import { Layers3, Map as MapIcon, Ruler } from 'lucide-react'
import { LAYERS, useApp } from '@/context/app-state'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Switch } from '@/components/ui/switch'
import { Dot } from '@/features/workspace/common'

export type Basemap = 'light' | 'dark' | 'imagery' | 'osm'
export type MeasureMode = 'none' | 'distance' | 'area'

const BASEMAP_LABEL: Record<Basemap, string> = { light: 'Light', dark: 'Dark', imagery: 'Imagery', osm: 'Streets' }

export function MapToolbar({
  basemap,
  setBasemap,
  measure,
  setMeasure,
}: {
  basemap: Basemap
  setBasemap: (b: Basemap) => void
  measure: MeasureMode
  setMeasure: (m: MeasureMode) => void
}) {
  const app = useApp()
  const groups = [...new Set(LAYERS.map((l) => l.group))]
  return (
    <div className='flex items-center gap-0.5 rounded-lg border bg-background/95 p-0.5 shadow-sm'>
      <Popover>
        <PopoverTrigger asChild>
          <Button variant='ghost' size='sm' className='h-8 gap-1.5 px-2.5 text-xs' aria-label='Layers'>
            <Layers3 className='size-3.5' /> <span className='hidden @2xl:inline'>Layers</span>
          </Button>
        </PopoverTrigger>
        <PopoverContent align='end' className='w-64 p-0'>
          {groups.map((g) => (
            <div key={g} className='border-b p-2 last:border-b-0'>
              <div className='px-1 pb-1 text-[0.6875rem] font-medium text-muted-foreground'>{g}</div>
              {LAYERS.filter((l) => l.group === g).map((l) => (
                <label key={l.id} className='flex cursor-pointer items-center gap-2 rounded px-1 py-1 text-xs hover:bg-muted/60'>
                  <Dot color={l.color} />
                  <span className='flex-1'>{l.label}</span>
                  <Switch checked={app.visible[l.id]} onCheckedChange={(v) => app.toggle(l.id, v)} className='scale-75' />
                </label>
              ))}
            </div>
          ))}
        </PopoverContent>
      </Popover>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant='ghost' size='sm' className='h-8 gap-1.5 px-2.5 text-xs' aria-label='Background map'>
            <MapIcon className='size-3.5' /> <span className='hidden @2xl:inline'>{BASEMAP_LABEL[basemap]}</span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align='end' className='w-56'>
          <DropdownMenuLabel className='text-[0.6875rem] font-normal text-muted-foreground'>Background map (display only)</DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuRadioGroup value={basemap} onValueChange={(v) => setBasemap(v as Basemap)}>
            {(Object.keys(BASEMAP_LABEL) as Basemap[]).map((b) => (
              <DropdownMenuRadioItem key={b} value={b} className='text-xs'>
                {BASEMAP_LABEL[b]}
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
        </DropdownMenuContent>
      </DropdownMenu>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant={measure === 'none' ? 'ghost' : 'secondary'} size='sm' className='h-8 gap-1.5 px-2.5 text-xs' aria-label='Measure'>
            <Ruler className='size-3.5' /> <span className='hidden @2xl:inline'>Measure</span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align='end'>
          <DropdownMenuRadioGroup value={measure} onValueChange={(v) => setMeasure(v as MeasureMode)}>
            <DropdownMenuRadioItem value='distance' className='text-xs'>
              Distance
            </DropdownMenuRadioItem>
            <DropdownMenuRadioItem value='area' className='text-xs'>
              Area
            </DropdownMenuRadioItem>
            <DropdownMenuRadioItem value='none' className='text-xs'>
              Off
            </DropdownMenuRadioItem>
          </DropdownMenuRadioGroup>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}
