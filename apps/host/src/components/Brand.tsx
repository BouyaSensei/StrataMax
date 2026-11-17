import { tr } from '../lib/i18n';
/** The approved three-bar mark and outlined wordmark, sized as artwork. */
import { useEffect, useState, type CSSProperties } from 'react';
import { useMedia } from '../lib/responsive';
import { useStore } from '../lib/store';
export function Brand({ variant, height, decorative = false, className = '', }: {
    variant: 'mark' | 'lockup';
    height: number;
    decorative?: boolean;
    className?: string;
}) {
    const preference = useStore(s => s.theme);
    const systemDark = useMedia('(prefers-color-scheme: dark)');
    const theme = preference === 'system' ? (systemDark ? 'dark' : 'light') : preference;
    const src = `/brand/strata-${variant}-${theme}.svg`;
    const [failed, setFailed] = useState(false);
    useEffect(() => setFailed(false), [src]);
    // StrataMax: l'art du wordmark reste celui de Strata, et le suffixe "Max" est pose a cote pour
    // distinguer le fork de l'application de base. L'identifiant, les ports et les dossiers de donnees
    // ne bougent pas : seul l'affichage change.
    const suffix = variant === 'lockup' ? 'Max' : null;
    const width = variant === 'mark' ? height : height * 182 / 48;
    const style = { width: suffix ? Math.round(width + height * 0.98) : Math.round(width), height } satisfies CSSProperties;
    return <span className={`brand brand--${variant} ${className}`} style={style} role={decorative ? undefined : 'img'} aria-label={tr(decorative ? undefined : 'StrataMax')} aria-hidden={decorative || undefined}>
    {tr(!failed ? <img className="brand__img" src={src} width={Math.round(width)} height={height} style={{ width: Math.round(width) }} alt={tr("")} onError={() => setFailed(true)}/> : null)}
    {suffix ? <span className="brand__max" style={{ fontSize: Math.max(9, Math.round(height * 0.42)) }} aria-hidden="true">{tr(suffix)}</span> : null}
    <span className="brand__fallback" aria-hidden="true">{tr(variant === 'mark' ? 'S' : 'StrataMax')}</span>
  </span>;
}
