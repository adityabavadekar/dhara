import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { Check, Copy, Loader2, Send, X } from 'lucide-react'
import { useApp } from '@/context/app-state'
import { api } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'

type Entry = { id: string; role: 'user' | 'assistant'; content: string; error?: boolean }

// These exact strings have verified template answers on the server.
const SUGGESTED = [
  'Provide an executive situation briefing for disaster commanders.',
  'Which settlements are completely cut off without road access?',
  'Which bridges are destroyed or impassable across river crossings?',
  'Which health facilities cannot be reached by road?',
  'What is the top priority road or bridge repair to reconnect communities?',
  'What are the longest hospital evacuation detours caused by road severances?',
  'How many homes and structures are exposed in the high-hazard zone?',
  'Which areas could not be observed due to cloud cover or radar shadow?',
]

const uid = () => Math.random().toString(36).slice(2, 9)

export function AssistantPanel() {
  const app = useApp()
  const id = app.jobId
  const [lang, setLang] = useState<'en' | 'ne'>('en')
  const [prompt, setPrompt] = useState('')
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState<Entry[]>([])
  const [copied, setCopied] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [history, loading])

  // Conversations belong to one analysis.
  useEffect(() => setHistory([]), [id])

  const ask = async (text: string) => {
    if (!id || !text.trim() || loading) return
    const q = text.trim()
    setHistory((h) => [...h, { id: uid(), role: 'user', content: q }])
    setPrompt('')
    setLoading(true)
    try {
      const res = await api.chat(id, q, lang)
      setHistory((h) => [...h, { id: uid(), role: 'assistant', content: res.answer }])
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
      askRef.current(app.assistantDraft)
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
            <p className='pb-1 text-[0.6875rem] text-muted-foreground'>Suggested</p>
            {SUGGESTED.map((q) => (
              <button
                key={q}
                type='button'
                onClick={() => ask(q)}
                className='block w-full rounded-md border bg-card px-2.5 py-2 text-left text-xs hover:bg-muted/60'
              >
                {q}
              </button>
            ))}
          </div>
        )}
        {history.map((m) => (
          <div key={m.id} className={cn('flex flex-col gap-1', m.role === 'user' ? 'items-end' : 'items-start')}>
            <div
              className={cn(
                'max-w-[92%] rounded-lg px-3 py-2 text-xs leading-relaxed',
                m.role === 'user' ? 'bg-primary text-primary-foreground' : 'border bg-card',
                m.error && 'border-destructive/40 text-destructive'
              )}
            >
              {m.role === 'user' ? (
                <div className='whitespace-pre-wrap'>{m.content}</div>
              ) : (
                <div className='space-y-1.5 [&_h2]:text-sm [&_h2]:font-semibold [&_h3]:text-xs [&_h3]:font-semibold [&_li]:my-0.5 [&_ol]:list-decimal [&_ol]:pl-4 [&_strong]:font-semibold [&_ul]:list-disc [&_ul]:pl-4'>
                  <ReactMarkdown>{m.content}</ReactMarkdown>
                </div>
              )}
            </div>
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
          </div>
        ))}
        {loading && (
          <div className='flex items-center gap-2 text-xs text-muted-foreground'>
            <Loader2 className='size-3.5 animate-spin' /> Checking the results…
          </div>
        )}
      </div>

      {id && history.length > 0 && (
        <div className='flex gap-1.5 overflow-x-auto border-t px-2.5 pt-2 pb-0.5 no-scrollbar'>
          {SUGGESTED.filter((q) => !history.some((h) => h.role === 'user' && h.content === q)).map((q) => (
            <button
              key={q}
              type='button'
              disabled={loading}
              onClick={() => ask(q)}
              className='shrink-0 rounded-full border bg-card px-2.5 py-1 text-[0.6875rem] whitespace-nowrap text-muted-foreground hover:bg-muted/60 hover:text-foreground disabled:opacity-50'
            >
              {q}
            </button>
          ))}
        </div>
      )}
      <form
        onSubmit={(e) => {
          e.preventDefault()
          ask(prompt)
        }}
        className={cn('flex items-center gap-1.5 p-2.5', history.length === 0 && 'border-t')}
      >
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
      </form>
    </aside>
  )
}
