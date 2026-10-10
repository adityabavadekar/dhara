import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import 'maplibre-gl/dist/maplibre-gl.css'
import './styles/index.css'
import { App } from '@/app'
import { AppStateProvider } from '@/context/app-state'
import { ThemeProvider } from '@/context/theme-provider'
import { Toaster } from '@/components/ui/sonner'
import { TooltipProvider } from '@/components/ui/tooltip'

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } })

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <TooltipProvider delayDuration={200}>
          <AppStateProvider>
            <App />
            <Toaster position='bottom-right' />
          </AppStateProvider>
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  </StrictMode>
)
