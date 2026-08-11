import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** shadcn's class helper: conditional classes via clsx, then Tailwind
 *  conflict resolution via tailwind-merge so a caller's `px-2` beats a
 *  component's built-in `px-1` instead of both landing in the class list and
 *  the outcome depending on stylesheet order. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
