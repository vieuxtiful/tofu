/** The short word that rides beside a label to qualify it.
 *
 * One component rather than a class string copied per site. There are now
 * four of these — `new` on recommendations, `new` on the capture review
 * banner, `active` on the asset list, `best` on the Guided capture mode —
 * and they are the same idiom: a small violet word that says something about
 * the thing next to it without becoming a control. Four hand-written copies
 * had already drifted to three different weights.
 *
 * Deliberately not a badge with a background. These sit inside dense rows of
 * status text, and a filled chip would outrank the label it qualifies.
 */
export default function StatusFlag({
  children, className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span
      className={`shrink-0 normal-case text-[10px] font-medium tracking-normal text-violet-600 dark:text-violet-300 ${className}`}
      data-testid="status-flag"
    >
      {children}
    </span>
  );
}
