export function normalizeMetadata(value: any): any {
  if (Array.isArray(value)) {
    return value.map(normalizeMetadata);
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, val]) => [key, normalizeMetadata(val)])
    );
  }
  if (value === "" || value === undefined) return null;
  return value;
}
