import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

type RouteContext = {
  params: Promise<{
    path: string[];
  }>;
};

const INTERNAL_API_BASE = process.env.CUPCAST_API_INTERNAL_BASE?.trim() || "http://127.0.0.1:8000";
const API_START_HINT =
  "Run `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_api_dev.ps1` from the CupCast AI repo root and keep that terminal open.";

export async function GET(request: NextRequest, context: RouteContext) {
  return proxyCupcastRequest(request, context);
}

export async function POST(request: NextRequest, context: RouteContext) {
  return proxyCupcastRequest(request, context);
}

async function proxyCupcastRequest(request: NextRequest, context: RouteContext): Promise<Response> {
  const { path } = await context.params;
  const target = buildTargetUrl(path, request.nextUrl.search);

  try {
    const upstream = await fetch(target, {
      method: request.method,
      headers: forwardedHeaders(request),
      body: request.method === "GET" || request.method === "HEAD" ? undefined : await request.arrayBuffer(),
      cache: "no-store"
    });

    return new Response(await upstream.arrayBuffer(), {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: responseHeaders(upstream.headers)
    });
  } catch {
    return NextResponse.json(
      {
        detail: `Unable to reach CupCast API at ${INTERNAL_API_BASE}. ${API_START_HINT}`
      },
      { status: 503 }
    );
  }
}

function buildTargetUrl(path: string[], search: string): string {
  const base = INTERNAL_API_BASE.endsWith("/") ? INTERNAL_API_BASE : `${INTERNAL_API_BASE}/`;
  const target = new URL(path.map(encodeURIComponent).join("/"), base);
  target.search = search;
  return target.toString();
}

function forwardedHeaders(request: NextRequest): Headers {
  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  const accept = request.headers.get("accept");

  if (contentType) headers.set("content-type", contentType);
  if (accept) headers.set("accept", accept);

  return headers;
}

function responseHeaders(headers: Headers): Headers {
  const nextHeaders = new Headers(headers);
  nextHeaders.delete("content-encoding");
  nextHeaders.delete("content-length");
  nextHeaders.delete("transfer-encoding");
  return nextHeaders;
}
