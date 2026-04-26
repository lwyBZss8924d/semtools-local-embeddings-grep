use crate::jina::types::{EmbeddingBackend, JinaTask, PromptName};
use crate::jina_runtime::NativeJinaEmbedder;
use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum DaemonRequest {
    Ping,
    Shutdown,
    Embed(EmbedRequest),
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbedRequest {
    texts: Vec<String>,
    model: String,
    task: String,
    prompt_name: Option<String>,
    truncate_dim: Option<usize>,
    model_dir: Option<String>,
    batch_size: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbedResponse {
    ok: bool,
    embeddings: Option<Vec<Vec<f32>>>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DaemonStatus {
    pub running: bool,
    pub socket_path: String,
    pub pid_path: String,
    pub pid: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct DaemonJinaClient {
    model_dir: Option<PathBuf>,
    batch_size: usize,
}

impl DaemonJinaClient {
    pub fn new(model_dir: Option<PathBuf>, batch_size: usize) -> Self {
        Self {
            model_dir,
            batch_size: batch_size.max(1),
        }
    }
}

impl EmbeddingBackend for DaemonJinaClient {
    fn embed(
        &self,
        texts: &[String],
        model: &str,
        task: &JinaTask,
        prompt_name: Option<PromptName>,
        truncate_dim: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let request = DaemonRequest::Embed(EmbedRequest {
            texts: texts.to_vec(),
            model: model.to_string(),
            task: task.to_string(),
            prompt_name: prompt_name.map(|prompt| prompt.as_wire().to_string()),
            truncate_dim,
            model_dir: self
                .model_dir
                .as_ref()
                .map(|path| path.display().to_string()),
            batch_size: self.batch_size,
        });
        let response = send_request(&request)
            .context("failed to call semtools jgrep daemon; run `semtools jgrep --daemon-start`")?;
        if response.ok {
            response
                .embeddings
                .ok_or_else(|| anyhow!("daemon returned no embeddings"))
        } else {
            Err(anyhow!(
                "{}",
                response
                    .error
                    .unwrap_or_else(|| "unknown daemon error".to_string())
            ))
        }
    }
}

pub fn start_background() -> Result<DaemonStatus> {
    let existing = status();
    if existing.running {
        return Ok(existing);
    }

    cleanup_stale_socket()?;
    let exe = std::env::current_exe()?;
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path()?)?;
    let err = log.try_clone()?;
    Command::new(exe)
        .arg("jgrep")
        .arg("--daemon-serve-foreground")
        .process_group(0)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(err))
        .spawn()
        .context("failed to spawn semtools jgrep daemon")?;

    for _ in 0..100 {
        std::thread::sleep(Duration::from_millis(100));
        let status = status();
        if status.running {
            return Ok(status);
        }
    }

    bail!("daemon did not become ready; see {}", log_path()?.display())
}

pub fn status() -> DaemonStatus {
    let socket = socket_path().unwrap_or_else(|_| PathBuf::from(""));
    let pid_path = pid_path().unwrap_or_else(|_| PathBuf::from(""));
    let pid = read_pid(&pid_path);
    let running = send_request(&DaemonRequest::Ping)
        .map(|response| response.ok)
        .unwrap_or(false);
    DaemonStatus {
        running,
        socket_path: socket.display().to_string(),
        pid_path: pid_path.display().to_string(),
        pid,
    }
}

pub fn stop() -> Result<DaemonStatus> {
    let _ = send_request(&DaemonRequest::Shutdown);
    for _ in 0..50 {
        std::thread::sleep(Duration::from_millis(100));
        let status = status();
        if !status.running {
            cleanup_stale_socket()?;
            return Ok(status);
        }
    }
    bail!("daemon did not stop")
}

pub fn serve_foreground() -> Result<()> {
    let socket = socket_path()?;
    cleanup_stale_socket()?;
    if let Some(parent) = socket.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(pid_path()?, std::process::id().to_string())?;
    let listener = UnixListener::bind(&socket)
        .with_context(|| format!("failed to bind {}", socket.display()))?;

    for stream in listener.incoming() {
        let stream = stream?;
        match handle_stream(stream) {
            Ok(true) => break,
            Ok(false) => {}
            Err(err) => {
                eprintln!("daemon request failed: {err}");
            }
        }
    }

    cleanup_stale_socket()?;
    let _ = std::fs::remove_file(pid_path()?);
    Ok(())
}

fn handle_stream(mut stream: UnixStream) -> Result<bool> {
    let mut line = String::new();
    {
        let mut reader = BufReader::new(&stream);
        reader.read_line(&mut line)?;
    }
    let request: DaemonRequest = serde_json::from_str(line.trim())?;
    match request {
        DaemonRequest::Ping => {
            write_response(&mut stream, &EmbedResponse::ok(Vec::new()))?;
            Ok(false)
        }
        DaemonRequest::Shutdown => {
            write_response(&mut stream, &EmbedResponse::ok(Vec::new()))?;
            Ok(true)
        }
        DaemonRequest::Embed(request) => {
            let response = match embed_request(request) {
                Ok(embeddings) => EmbedResponse::ok(embeddings),
                Err(err) => EmbedResponse::err(err.to_string()),
            };
            write_response(&mut stream, &response)?;
            Ok(false)
        }
    }
}

fn embed_request(request: EmbedRequest) -> Result<Vec<Vec<f32>>> {
    let task = JinaTask::from_wire(&request.task)
        .ok_or_else(|| anyhow!("unsupported task: {}", request.task))?;
    let prompt_name = request
        .prompt_name
        .as_deref()
        .map(|value| {
            PromptName::from_wire(value)
                .ok_or_else(|| anyhow!("unsupported prompt_name: {}", value))
        })
        .transpose()?;
    let embedder = NativeJinaEmbedder::new_with_batch_size(
        request.model_dir.map(PathBuf::from),
        request.batch_size,
    );
    embedder.embed(
        &request.texts,
        &request.model,
        &task,
        prompt_name,
        request.truncate_dim,
    )
}

fn send_request(request: &DaemonRequest) -> Result<EmbedResponse> {
    let mut stream = UnixStream::connect(socket_path()?)?;
    let line = serde_json::to_string(request)?;
    stream.write_all(line.as_bytes())?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    let mut reader = BufReader::new(stream);
    let mut response = String::new();
    reader.read_line(&mut response)?;
    Ok(serde_json::from_str(response.trim())?)
}

fn write_response(stream: &mut UnixStream, response: &EmbedResponse) -> Result<()> {
    let line = serde_json::to_string(response)?;
    stream.write_all(line.as_bytes())?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    Ok(())
}

impl EmbedResponse {
    fn ok(embeddings: Vec<Vec<f32>>) -> Self {
        Self {
            ok: true,
            embeddings: Some(embeddings),
            error: None,
        }
    }

    fn err(error: String) -> Self {
        Self {
            ok: false,
            embeddings: None,
            error: Some(error),
        }
    }
}

fn runtime_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("No home dir found"))?;
    let dir = home.join(".semtools").join("runtime");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn socket_path() -> Result<PathBuf> {
    Ok(runtime_dir()?.join("jina-daemon.sock"))
}

fn pid_path() -> Result<PathBuf> {
    Ok(runtime_dir()?.join("jina-daemon.pid"))
}

fn log_path() -> Result<PathBuf> {
    Ok(runtime_dir()?.join("jina-daemon.log"))
}

fn read_pid(path: &Path) -> Option<u32> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|content| content.trim().parse::<u32>().ok())
}

fn cleanup_stale_socket() -> Result<()> {
    let socket = socket_path()?;
    if socket.exists() && UnixStream::connect(&socket).is_err() {
        std::fs::remove_file(socket)?;
    }
    Ok(())
}
