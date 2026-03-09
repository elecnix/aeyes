use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("AEYES_DAEMON").ok().as_deref() == Some("1") {
        aeyes::run_daemon_from_env().await
    } else {
        aeyes::run_cli().await
    }
}
