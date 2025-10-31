export async function readBufferToString(buf: Buffer): Promise<string> {
return buf.toString("utf-8");
}