from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.main import mount_frontend


def test_serves_spa_with_fallback_and_blocks_traversal(tmp_path):
    site = tmp_path / "dist"
    (site / "assets").mkdir(parents=True)
    (site / "index.html").write_text("<html>app</html>")
    (site / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("nope")

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    mount_frontend(app, str(site))
    client = TestClient(app)

    assert client.get("/health").json() == {"status": "ok"}  # API routes still win
    assert client.get("/assets/app.js").text == "console.log(1)"
    assert "app" in client.get("/").text
    assert "app" in client.get("/some/client/route").text  # SPA fallback
    assert "nope" not in client.get("/..%2Fsecret.txt").text
