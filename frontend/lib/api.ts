export const API_BASE = process.env.NEXT_PUBLIC_CUPCAST_API_BASE?.trim() || "/api/cupcast";
const API_START_HINT =
  "Run `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_api_dev.ps1` from the CupCast AI repo root and keep that terminal open.";

export async function postJson<TResponse>(path: string, body: unknown): Promise<TResponse> {
  return requestJson<TResponse>(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body)
  });
}

export async function getJson<TResponse>(path: string): Promise<TResponse> {
  return requestJson<TResponse>(path);
}

async function requestJson<TResponse>(path: string, init?: RequestInit): Promise<TResponse> {
  const url = `${API_BASE}${path}`;
  let response: Response;

  try {
    response = await fetch(url, { ...init, cache: "no-store" });
  } catch {
    throw new Error(`Unable to reach CupCast API through ${API_BASE}. ${API_START_HINT}`);
  }

  if (!response.ok) {
    throw new Error(`API request failed: ${response.status}${await responseDetail(response)}`);
  }
  return response.json() as Promise<TResponse>;
}

async function responseDetail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    return typeof body.detail === "string" ? ` - ${body.detail}` : "";
  } catch {
    return "";
  }
}
