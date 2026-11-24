'use client';

import { usePipecatClientTransportState } from '@pipecat-ai/client-react';
import { ChevronDownIcon, LanguagesIcon } from 'lucide-react';

import { Button } from '@/components/ui/button';
import {
  Popover, PopoverContent, PopoverDescription, PopoverTitle, PopoverTrigger,
} from '@/components/ui/popover';
import { TWI_DIALECTS, type TwiDialect } from '@/lib/twi-dialects';

interface DialectSelectProps {
  dialect: TwiDialect;
  onDialectChange: (dialect: TwiDialect) => void;
  isConnecting: boolean;
}

export function DialectSelect({ dialect, onDialectChange, isConnecting }: DialectSelectProps) {
  const state = usePipecatClientTransportState();
  const locked = isConnecting || !['initialized', 'disconnected', 'error'].includes(state);
  const selected = TWI_DIALECTS.find((option) => option.id === dialect)!;

  return (
    <Popover>
      <PopoverTrigger
        render={<Button variant="outline" size="sm" />}
        aria-label={`Twi dialect: ${selected.label}`}
        className="min-w-0 gap-2"
      >
        <LanguagesIcon className="size-4 shrink-0" />
        <span>{selected.shortLabel}<span className="hidden sm:inline"> Twi</span></span>
        <ChevronDownIcon className="size-3 shrink-0 text-muted-foreground" />
      </PopoverTrigger>
      <PopoverContent align="end" className="w-72 max-w-[calc(100vw-1.5rem)] gap-3 p-4">
        <div className="space-y-1">
          <PopoverTitle>Choose your Twi dialect</PopoverTitle>
          <PopoverDescription>Guide the wording and pronunciation of Twi replies.</PopoverDescription>
        </div>
        <fieldset disabled={locked} aria-describedby="dialect-help" className="space-y-2">
          <legend className="sr-only">Twi dialect</legend>
          {TWI_DIALECTS.map((option) => (
            <label
              key={option.id}
              className={`flex items-center gap-3 rounded-xl border p-3 has-focus-visible:ring-2 has-focus-visible:ring-ring ${
                dialect === option.id ? 'border-primary/50 bg-primary/5' : 'border-border hover:bg-accent/50'
              } ${locked ? 'cursor-not-allowed opacity-65' : 'cursor-pointer'}`}
            >
              <input
                type="radio"
                name="twi-dialect"
                value={option.id}
                checked={dialect === option.id}
                onChange={() => onDialectChange(option.id)}
                className="size-4 accent-primary"
              />
              <span className="font-medium">{option.label}</span>
            </label>
          ))}
        </fieldset>
        <p id="dialect-help" role="status" className="text-xs text-muted-foreground">
          {locked
            ? `Current dialect: ${selected.label}. End the session to change it.`
            : `${selected.label} will apply when you connect. English is still available.`}
        </p>
      </PopoverContent>
    </Popover>
  );
}
