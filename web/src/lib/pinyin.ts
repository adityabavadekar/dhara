// pinyin-pro is large (about 140 KB compressed) and only needed for Chinese-script names, which few
// analyses contain, so it is loaded on first use instead of with the app.
import { useEffect, useState } from 'react'

type Pinyin = typeof import('pinyin-pro').pinyin
let fn: Pinyin | null = null
let loading: Promise<Pinyin> | null = null

const HAN = /\p{Script=Han}/u
export const hasHan = (s: string) => HAN.test(s)

/** The pinyin function once loaded, else null (callers fall back to the plain name). */
export const pinyinNow = (): Pinyin | null => fn

export function loadPinyin(): Promise<Pinyin> {
  loading ??= import('pinyin-pro').then((m) => (fn = m.pinyin))
  return loading
}

/** True once pinyin is available; starts loading when `needed` (e.g. a Chinese name is on screen). */
export function usePinyinReady(needed: boolean): boolean {
  const [ready, setReady] = useState(fn !== null)
  useEffect(() => {
    if (needed && !ready) loadPinyin().then(() => setReady(true))
  }, [needed, ready])
  return ready
}
