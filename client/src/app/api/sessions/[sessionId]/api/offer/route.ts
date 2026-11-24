import { NextResponse } from 'next/server';

type Context = { params: Promise<{ sessionId: string }> };

async function forwardOffer(request: Request, context: Context) {
  const { sessionId } = await context.params;
  const startUrl = process.env.BOT_START_URL || 'http://localhost:7860/start';
  const offerUrl = new URL(startUrl);
  offerUrl.pathname = offerUrl.pathname.replace(
    /\/start\/?$/,
    `/sessions/${encodeURIComponent(sessionId)}/api/offer`,
  );
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (process.env.BOT_START_PUBLIC_API_KEY) {
    headers.Authorization = `Bearer ${process.env.BOT_START_PUBLIC_API_KEY}`;
  }
  try {
    const response = await fetch(offerUrl, {
      method: request.method,
      headers,
      body: await request.text(),
      signal: request.signal,
    });
    return new Response(response.body, {
      status: response.status,
      headers: { 'Content-Type': response.headers.get('Content-Type') || 'application/json' },
    });
  } catch {
    return NextResponse.json({ error: 'Unable to reach the bot server' }, { status: 502 });
  }
}

export const POST = forwardOffer;
export const PATCH = forwardOffer;
