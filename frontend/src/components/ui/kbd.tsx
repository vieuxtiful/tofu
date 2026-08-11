import { cn } from "@/lib/utils";

/*
 * Vendored from shadcn/ui — registry item `kbd`, new-york style, fetched
 * 2026-08-11 from https://ui.shadcn.com/r/styles/new-york/kbd.json.
 *
 * Vendored rather than installed via `npx shadcn@latest add kbd`, which
 * requires `npx shadcn@latest init`, which rewrites index.css, tsconfig.json
 * and the Tailwind config. This project's index.css carries a deliberate
 * history (daisyUI removed on purpose, the Tailwind v4 border-colour
 * compatibility layer, an explicit font stack chosen for Vietnamese stacked
 * diacritics) and uikit.css is a large bespoke system. Letting the
 * initializer rewrite both to obtain one keycap was the wrong trade.
 *
 * ONE DELIBERATE DIVERGENCE from the registry source: the upstream classes
 * `bg-muted text-muted-foreground` reference shadcn theme tokens that this
 * project never defines, so verbatim they render an invisible keycap. They
 * are replaced with the zinc pairing every other surface here already uses.
 * Structure, API, `data-slot` attributes and the rest of the class list are
 * untouched, so a future re-add diffs to almost nothing.
 */

function Kbd({ className, ...props }: React.ComponentProps<"kbd">) {
  return (
    <kbd
      data-slot="kbd"
      className={cn(
        "bg-zinc-200 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300",
        "pointer-events-none inline-flex h-5 w-fit min-w-5 select-none items-center justify-center gap-1 rounded-sm px-1 font-sans text-xs font-medium",
        "[&_svg:not([class*='size-'])]:size-3",
        "[[data-slot=tooltip-content]_&]:bg-background/20 [[data-slot=tooltip-content]_&]:text-background dark:[[data-slot=tooltip-content]_&]:bg-background/10",
        className,
      )}
      {...props}
    />
  );
}

function KbdGroup({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <kbd
      data-slot="kbd-group"
      className={cn("inline-flex items-center gap-1", className)}
      {...props}
    />
  );
}

export { Kbd, KbdGroup };
