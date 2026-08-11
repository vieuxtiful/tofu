/** Outline keyboard glyph.
 *
 * Supplied as two components, one defaulting to #000000 and one to #FFFFFF
 * for light and dark. They were otherwise byte-identical, and the split
 * would not have worked: the `<path>` fills with `currentColor`, so the
 * `color` prop only ever reached `stroke` — and an outline path with no
 * stroked geometry ignores it. Both variants therefore rendered in the
 * inherited text colour regardless of their default.
 *
 * So: one component that inherits `currentColor`, like every other icon in
 * this app. Dark mode is handled by whatever class chain the caller already
 * uses for its text, which is also how it stays correct if the palette
 * changes.
 */
export default function KeyboardIcon({
  size = 14,
  className,
  ...props
}: React.ComponentProps<"svg"> & { size?: number }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      aria-hidden="true"
      focusable="false"
      className={className}
      {...props}
    >
      <path
        fill="currentColor"
        d="M4 5a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm0 2h16v10H4zm1 1v2h2V8zm3 0v2h2V8zm3 0v2h2V8zm3 0v2h2V8zm3 0v2h2V8zM5 11v2h2v-2zm3 0v2h2v-2zm3 0v2h2v-2zm3 0v2h2v-2zm3 0v2h2v-2zm-9 3v2h8v-2z"
      />
    </svg>
  );
}
