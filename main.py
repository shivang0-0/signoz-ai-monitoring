from fastapi import FastAPI

app = FastAPI(title="SigNoz AI Monitoring")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
