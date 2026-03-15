/**
 * Shared utility functions.
 */

/** Converts a snake_case key into a human-readable label. */
export function formatKey(key: string): string {
  return key.split("_").join(" ");
}
