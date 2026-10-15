import { useEffect, useRef, useState } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  ArrowUpRight,
  Building,
  Check,
  ChevronDown,
  ClipboardList,
  Construction,
  Copy,
  EyeOff,
  Hospital,
  Info,
  Loader2,
  MapPin,
  Route,
  RotateCcw,
  Send,
  ShieldAlert,
  Square,
  Volume2,
  Wrench,
  X,
  type LucideIcon,
} from 'lucide-react'
import { useApp } from '@/context/app-state'
import { useQuery } from '@tanstack/react-query'
import { api, type ChatCard } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'

type EntityContext = { id: string; kind: string; coordinates: [number, number] }
type Entry = {
  id: string
  role: 'user' | 'assistant'
  content: string
  entity?: EntityContext
  cards?: ChatCard[]
  llmGenerated?: boolean
  error?: boolean
}

// Suggested questions come from the server with their answers already computed from the job's
// facts, so a click shows the answer at once and never goes to a language model.
const SUGGESTED_ICONS: Record<string, LucideIcon> = {
  briefing: ClipboardList,
  cut_off: MapPin,
  bridges: Construction,
  health: Hospital,
  repair: Wrench,
  detours: Route,
  exposure: Building,
  unobserved: EyeOff,
}

const uid = () => Math.random().toString(36).slice(2, 9)
const SCENE_ID = /^S[12][A-D]_.*(?:COG)?$/

export function AssistantPanel() {
  const app = useApp()
  const id = app.jobId
  const [lang, setLang] = useState<'en' | 'ne'>('en')
  // Read aloud with the browser's own speech synthesis (on the device; nothing is sent anywhere).
  // Nepali uses a Nepali voice when installed, else a Hindi one (same Devanagari script).
  const canSpeak = typeof window !== 'undefined' && 'speechSynthesis' in window
  const [speaking, setSpeaking] = useState<string | null>(null)
  const stopSpeaking = () => {
    window.speechSynthesis.cancel()
    setSpeaking(null)
  }
  const speak = (msgId: string, text: string) => {
    const synth = window.speechSynthesis
    synth.cancel()
    const plain = text.replace(/[#*_`>|]/g, '').replace(/\[(.*?)\]\(.*?\)/g, '$1')
    const u = new SpeechSynthesisUtterance(plain)
    const devanagari = /[\u0900-\u097F]/.test(plain)
    const voices = synth.getVoices()
    const voice = devanagari ? (voices.find((v) => v.lang.startsWith('ne')) ?? voices.find((v) => v.lang.startsWith('hi'))) : voices.find((v) => v.lang.startsWith('en'))
    if (voice) u.voice = voice
    u.lang = voice?.lang ?? (devanagari ? 'ne-NP' : 'en-US')
    u.onend = u.onerror = () => setSpeaking(null)
    setSpeaking(msgId)
    synth.speak(u)
  }
  const [prompt, setPrompt] = useState('')
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState<Entry[]>([])
  const [copied, setCopied] = useState<string | null>(null)
  const [askAboutOpen, setAskAboutOpen] = useState(true)
  const [useMapContext, setUseMapContext] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const selection = app.selection
  const selectedEntity: EntityContext | undefined =
    selection && ['settlement', 'flow-settlement', 'bridge', 'hospital'].includes(selection.kind)
      ? { id: String(selection.props.osm_id ?? ''), kind: selection.kind, coordinates: selection.lngLat }
      : undefined
  const selectedEntityKey = selectedEntity ? `${selectedEntity.kind}:${selectedEntity.id}:${selectedEntity.coordinates.join(',')}` : ''
  const selectedEntityLabel = selection ? String(selection.props.name ?? selectedEntity?.id ?? 'Selected map feature') : ''
  const activeMapContext = useMapContext ? selectedEntity : undefined

  useEffect(() => {
    setUseMapContext(Boolean(selectedEntity))
  }, [selectedEntityKey])

  const markdownComponents: Components = {
    table: ({ children }) => (
      <div className='my-2 max-w-full overflow-x-auto rounded-md border'>
        <table className='w-max min-w-full border-collapse text-left text-xs'>{children}</table>
      </div>
    ),
    th: ({ children }) => <th className='whitespace-nowrap border-b bg-muted/50 px-2 py-1.5 font-semibold'>{children}</th>,
    td: ({ children }) => <td className='whitespace-nowrap border-b px-2 py-1.5 align-top last:border-b-0'>{children}</td>,
    code: ({ children, className, ...props }) => {
      const value = String(children).replace(/\n$/, '')
      if (SCENE_ID.test(value)) {
        const sceneCopyId = `scene:${value}`
        return (
          <span className='inline-flex w-max items-center gap-1 rounded border bg-muted/50 px-1.5 py-0.5 align-middle font-mono text-[0.6875rem]'>
            <span>{value}</span>
            <button
              type='button'
              className='shrink-0 rounded p-0.5 text-muted-foreground hover:bg-background hover:text-foreground'
              aria-label='Copy satellite scene ID'
              title='Copy scene ID'
              onClick={() => {
                navigator.clipboard.writeText(value)
                setCopied(sceneCopyId)
                setTimeout(() => setCopied(null), 1500)
              }}
            >
              {copied === sceneCopyId ? <Check className='size-3' /> : <Copy className='size-3' />}
            </button>
          </span>
        )
      }
      return (
        <code className={className} {...props}>
          {children}
        </code>
      )
    },
  }

  // Keep the latest question at the top, so the start of its answer is in view.
  useEffect(() => {
    const el = scrollRef.current
    const q = el?.querySelectorAll<HTMLElement>('[data-role="user"]')
    const last = q?.[q.length - 1]
    if (el && last) el.scrollTo({ top: el.scrollTop + last.getBoundingClientRect().top - el.getBoundingClientRect().top - 8 })
  }, [history, loading])

  // Conversations belong to one analysis.
  useEffect(() => {
    setHistory([])
  }, [id])

  const suggested = useQuery({
    queryKey: ['suggested', id, lang],
    queryFn: () => api.suggested(id!, lang),
    enabled: !!id && app.assistantOpen,
    staleTime: 5 * 60_000,
  })
  const showSuggested = (s: { question: string; answer: string; cards: ChatCard[] }) => {
    setAskAboutOpen(false)
    setHistory((h) => [...h, { id: uid(), role: 'user', content: s.question }, { id: uid(), role: 'assistant', content: s.answer, cards: s.cards, llmGenerated: false }])
  }

  const ask = async (text: string, entity?: EntityContext) => {
    if (!id || !text.trim() || loading) return
    const q = text.trim()
    setAskAboutOpen(false)
    setHistory((h) => [...h, { id: uid(), role: 'user', content: q, entity }])
    setPrompt('')
    setLoading(true)
    try {
      const res = await api.chat(id, q, lang, entity)
      setHistory((h) => [
        ...h,
        {
          id: uid(),
          role: 'assistant',
          content: res.answer,
          cards: res.cards,
          llmGenerated: res.provider === 'grounded_agent',
        },
      ])
    } catch (e) {
      setHistory((h) => [...h, { id: uid(), role: 'assistant', content: e instanceof Error ? e.message : String(e), error: true }])
    } finally {
      setLoading(false)
    }
  }

  const askRef = useRef(ask)
  askRef.current = ask
  useEffect(() => {
    if (app.assistantDraft && app.assistantOpen) {
      askRef.current(app.assistantDraft.question, app.assistantDraft.entity)
      app.clearAssistantDraft()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [app.assistantDraft, app.assistantOpen])

  if (!app.assistantOpen) return null

  return (
    <aside className='flex w-full shrink-0 flex-col border-l bg-background max-md:absolute max-md:inset-0 max-md:z-30 md:w-[23.75rem]'>
      <div className='flex items-center gap-2 border-b px-3 py-2.5'>
        <div className='min-w-0 flex-1'>
          <div className='text-sm font-semibold tracking-tight'>Assistant</div>
          <div className='text-[0.6875rem] text-muted-foreground'>Answers only from this analysis's results</div>
        </div>
        <div className='flex rounded-md border p-0.5'>
          {(['en', 'ne'] as const).map((l) => (
            <button
              key={l}
              type='button'
              onClick={() => setLang(l)}
              className={cn('rounded px-1.5 py-0.5 text-[0.6875rem] font-medium', lang === l ? 'bg-muted text-foreground' : 'text-muted-foreground')}
            >
              {l === 'en' ? 'EN' : 'नेपाली'}
            </button>
          ))}
        </div>
        <Button variant='ghost' size='icon' className='size-7' onClick={() => app.setAssistantOpen(false)} aria-label='Close'>
          <X className='size-4' />
        </Button>
      </div>

      <div ref={scrollRef} className='min-h-0 flex-1 space-y-3 overflow-y-auto p-3'>
        {!id && <p className='text-xs text-muted-foreground'>Open an analysis to ask about it.</p>}
        {id && history.length === 0 && (
          <div className='space-y-1'>
            <p className='pb-1 text-[0.6875rem] font-medium text-muted-foreground'>Try asking</p>
            <div className='grid grid-cols-2 gap-1.5'>
              {suggested.isLoading && <p className='col-span-2 text-xs text-muted-foreground'>Loading…</p>}
              {(suggested.data ?? []).map((s) => {
                const Icon = SUGGESTED_ICONS[s.id] ?? ClipboardList
                return (
                  <button
                    key={s.id}
                    type='button'
                    onClick={() => showSuggested(s)}
                    className='flex w-full items-start gap-2 rounded-md border bg-card px-2.5 py-2 text-left text-xs leading-snug hover:bg-muted/60'
                  >
                    <Icon className='mt-0.5 size-3.5 shrink-0 text-muted-foreground' aria-hidden='true' />
                    <span>{s.question}</span>
                  </button>
                )
              })}
            </div>
          </div>
        )}
        {history.map((m) => (
          <div key={m.id} data-role={m.role} className={cn('flex flex-col gap-1.5', m.role === 'user' ? 'items-end' : 'items-start')}>
            <div
              className={cn(
                'rounded-lg text-xs leading-relaxed',
                m.role === 'user' ? 'max-w-[92%] bg-primary px-3 py-2 text-primary-foreground' : 'w-full border bg-card px-3.5 py-3 shadow-sm',
                m.error && 'border-destructive/40 text-destructive',
              )}
            >
              {m.role === 'user' ? (
                <>
                  <div className='whitespace-pre-wrap'>{m.content}</div>
                  {m.entity && (
                    <div className='mt-1.5 border-t border-primary-foreground/20 pt-1.5 text-[0.625rem] leading-snug opacity-85'>
                      Map context · {m.entity.kind.replaceAll('-', ' ')}
                      {m.entity.id && ` · ${m.entity.id}`}
                      {` · ${m.entity.coordinates[1].toFixed(5)}°N, ${m.entity.coordinates[0].toFixed(5)}°E`}
                    </div>
                  )}
                </>
              ) : (
                <div className='space-y-2 text-[0.8125rem] leading-relaxed [&_h1]:mb-2 [&_h1]:text-base [&_h1]:font-semibold [&_h2]:mt-3 [&_h2]:border-b [&_h2]:pb-1.5 [&_h2]:text-sm [&_h2]:font-semibold [&_h3]:mt-2 [&_h3]:text-[0.8125rem] [&_h3]:font-semibold [&_li]:pl-0.5 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:space-y-1 [&_ol]:pl-5 [&_p]:my-1.5 [&_strong]:font-semibold [&_ul]:my-2 [&_ul]:list-disc [&_ul]:space-y-1 [&_ul]:pl-5 [&_blockquote]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:pl-3 [&_hr]:my-3 [&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-[0.75rem]'>
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                    {m.content}
                  </ReactMarkdown>
                </div>
              )}
              {m.role === 'assistant' && m.llmGenerated && (
                <p className='mt-3 flex items-start gap-1.5 border-t pt-2 text-[0.6875rem] leading-snug text-muted-foreground'>
                  <Info className='mt-0.5 size-3.5 shrink-0' aria-hidden='true' />
                  <span>
                    This response was generated by an LLM and grounded in this analysis’s available results. Accuracy may vary; verify important details against the results.
                  </span>
                </p>
              )}
            </div>
            {m.role === 'assistant' && !!m.cards?.length && (
              // Compact evidence chips (detail and source on hover) so the answer stays in view.
              <ul className='flex w-[92%] flex-wrap gap-1' aria-label='Analysis evidence'>
                {m.cards.map((card) =>
                  card.kind === 'limit' ? (
                    <li
                      key={card.key}
                      title={`${card.detail} · ${card.source}`}
                      className='flex w-full items-center gap-1.5 rounded border border-amber-500/30 bg-amber-500/[0.06] px-1.5 py-0.5 text-[0.625rem] leading-snug text-amber-900 dark:text-amber-200'
                    >
                      <ShieldAlert className='size-3 shrink-0' aria-hidden='true' />
                      <span className='min-w-0 truncate'>
                        {card.title}: {card.value}
                        {card.unit ? ` ${card.unit}` : ''}
                      </span>
                    </li>
                  ) : (
                    <li
                      key={card.key}
                      title={`${card.detail} · ${card.source}`}
                      className='inline-flex max-w-full items-baseline gap-1 rounded border bg-muted/40 px-1.5 py-0.5 text-[0.625rem] leading-snug'
                    >
                      <span className='truncate text-muted-foreground'>{card.title}</span>
                      <span className='font-semibold tabular-nums text-foreground'>{card.value}</span>
                      {card.unit && <span className='text-muted-foreground'>{card.unit}</span>}
                    </li>
                  ),
                )}
              </ul>
            )}
            {m.role === 'assistant' && !m.error && (
              <button
                type='button'
                onClick={() => {
                  navigator.clipboard.writeText(m.content)
                  setCopied(m.id)
                  setTimeout(() => setCopied(null), 1500)
                }}
                className='inline-flex items-center gap-1 px-1 text-[0.625rem] text-muted-foreground hover:text-foreground'
              >
                {copied === m.id ? <Check className='size-3' /> : <Copy className='size-3' />} Copy
              </button>
            )}
            {m.role === 'assistant' && !m.error && canSpeak && (
              <button
                type='button'
                onClick={() => (speaking === m.id ? stopSpeaking() : speak(m.id, m.content))}
                className='inline-flex items-center gap-1 px-1 text-[0.625rem] text-muted-foreground hover:text-foreground'
                aria-pressed={speaking === m.id}
              >
                {speaking === m.id ? <Square className='size-3' /> : <Volume2 className='size-3' />} {speaking === m.id ? 'Stop' : 'Listen'}
              </button>
            )}
          </div>
        ))}
        {loading && (
          <div className='flex items-center gap-2 text-xs text-muted-foreground'>
            <Loader2 className='size-3.5 animate-spin' /> Checking the results…
          </div>
        )}
      </div>

      {id && history.length > 0 && (
        <Collapsible open={askAboutOpen} onOpenChange={setAskAboutOpen} className={cn('border-t bg-muted/20 px-3', askAboutOpen ? 'pt-2.5 pb-2' : 'py-1')}>
          <div className={cn('flex items-center justify-between', askAboutOpen && 'mb-2')}>
            <CollapsibleTrigger className='inline-flex items-center gap-1.5 rounded-sm text-sm font-semibold tracking-tight hover:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'>
              <ChevronDown className={cn('size-4 transition-transform', !askAboutOpen && '-rotate-90')} aria-hidden='true' />
              <span>Ask about</span>
            </CollapsibleTrigger>
            <button
              type='button'
              disabled={loading}
              onClick={() => {
                setHistory([])
                setPrompt('')
              }}
              className='inline-flex items-center gap-1 rounded px-1.5 py-1 text-xs font-medium text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-50'
              aria-label='Clear conversation history'
              title='Clear conversation history'
            >
              <RotateCcw className='size-3.5' aria-hidden='true' /> Clear
            </button>
          </div>
          <CollapsibleContent>
            <div className='flex max-h-44 flex-col gap-1 overflow-y-auto pr-1'>
              {(suggested.data ?? [])
                .filter(({ question }) => !history.some((h) => h.role === 'user' && h.content === question))
                .map((s) => {
                  const Icon = SUGGESTED_ICONS[s.id] ?? ClipboardList
                  const question = s.question
                  return (
                    <button
                      key={s.id}
                      type='button'
                      disabled={loading}
                      onClick={() => showSuggested(s)}
                      className='flex w-full shrink-0 items-center justify-between gap-2 rounded-md border bg-background px-2.5 py-1.5 text-left text-xs leading-snug hover:bg-muted/60 disabled:opacity-50'
                    >
                      <span className='flex min-w-0 items-start gap-2'>
                        <Icon className='mt-0.5 size-3.5 shrink-0 text-muted-foreground' aria-hidden='true' />
                        <span>{question}</span>
                      </span>
                      <ArrowUpRight className='size-3.5 shrink-0 text-muted-foreground' aria-hidden='true' />
                    </button>
                  )
                })}
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}
      <form
        onSubmit={(e) => {
          e.preventDefault()
          ask(prompt, activeMapContext)
        }}
        className={cn('flex flex-col gap-1.5 p-2.5', history.length === 0 && 'border-t')}
      >
        {selectedEntity && (
          <div className='flex items-center justify-between gap-2 rounded-md border bg-muted/30 px-2 py-1 text-[0.6875rem]'>
            <span className='min-w-0 truncate text-muted-foreground'>
              {activeMapContext ? 'Question context' : 'Map feature available'} · {selectedEntity.kind.replaceAll('-', ' ')} · {selectedEntityLabel}
              {selectedEntity.id && ` · ${selectedEntity.id}`}
              {` · ${selectedEntity.coordinates[1].toFixed(5)}°N, ${selectedEntity.coordinates[0].toFixed(5)}°E`}
            </span>
            {activeMapContext ? (
              <button
                type='button'
                className='shrink-0 rounded p-0.5 text-muted-foreground hover:text-foreground'
                aria-label='Remove map context'
                title='Remove map context'
                onClick={() => setUseMapContext(false)}
              >
                <X className='size-3.5' />
              </button>
            ) : (
              <button type='button' className='shrink-0 rounded px-1 py-0.5 font-medium text-foreground hover:bg-muted' onClick={() => setUseMapContext(true)}>
                Add context
              </button>
            )}
          </div>
        )}
        <div className='flex items-center gap-1.5'>
          <Input
            className='h-8 flex-1 text-xs'
            placeholder={id ? 'Ask about damage, access or repairs' : 'Open an analysis first'}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={loading || !id}
          />
          <Button size='sm' className='h-8 px-3' disabled={loading || !prompt.trim() || !id} type='submit' aria-label='Send'>
            <Send className='size-3.5' />
          </Button>
        </div>
      </form>
    </aside>
  )
}
