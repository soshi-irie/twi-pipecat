'use client';

import { usePipecatClientTransportState } from '@pipecat-ai/client-react';
import { CheckIcon, ChevronDownIcon, AudioLinesIcon } from 'lucide-react';

import { Button } from '@/components/ui/button';
import {
  Popover,
  PopoverContent,
  PopoverDescription,
  PopoverTitle,
  PopoverTrigger,
} from '@/components/ui/popover';
import { VOICE_PROFILES, type VoiceProfile } from '@/lib/voice-profiles';

interface VoiceSelectProps {
  voice: VoiceProfile;
  onVoiceChange: (voice: VoiceProfile) => void;
  isConnecting: boolean;
}

export function VoiceSelect({ voice, onVoiceChange, isConnecting }: VoiceSelectProps) {
  const state = usePipecatClientTransportState();
  const locked = isConnecting || !['initialized', 'disconnected', 'error'].includes(state);

  return (
    <Popover>
      <PopoverTrigger
        render={<Button variant="outline" size="sm" />}
        aria-label={`Voice profile: ${voice}`}
        className="min-w-0 gap-2"
      >
        <AudioLinesIcon className="size-4 shrink-0" />
        <span className="hidden text-muted-foreground sm:inline">Voice</span>
        <span className="truncate">{voice}</span>
        <ChevronDownIcon className="size-3 shrink-0 text-muted-foreground" />
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 max-w-[calc(100vw-1.5rem)] gap-3 p-4">
        <div className="space-y-1">
          <PopoverTitle>Choose your voice</PopoverTitle>
          <PopoverDescription>
            A different personality for your conversation.
          </PopoverDescription>
        </div>
        <fieldset disabled={locked} aria-describedby="voice-help" className="space-y-2">
          <legend className="sr-only">Voice profile</legend>
          {VOICE_PROFILES.map((profile) => (
            <label
              key={profile.id}
              className={`relative flex cursor-pointer items-start gap-3 rounded-xl border p-3 transition-colors has-focus-visible:ring-2 has-focus-visible:ring-ring ${
                voice === profile.id
                  ? 'border-primary/50 bg-primary/5'
                  : 'border-border hover:bg-accent/50'
              } ${locked ? 'cursor-not-allowed opacity-65' : ''}`}
            >
              <input
                type="radio"
                name="voice-profile"
                value={profile.id}
                checked={voice === profile.id}
                onChange={() => onVoiceChange(profile.id)}
                className="sr-only"
              />
              <span className="flex-1 space-y-1">
                <span className="flex items-center justify-between gap-2">
                  <span className="font-medium">{profile.id}</span>
                  <span className="text-xs text-muted-foreground">{profile.gender}</span>
                </span>
                <span className="block text-xs font-medium text-primary">{profile.tone}</span>
                <span className="block text-xs text-muted-foreground">{profile.description}</span>
              </span>
              <span className={`mt-0.5 flex size-4 items-center justify-center rounded-full border ${
                voice === profile.id ? 'border-primary bg-primary text-primary-foreground' : 'border-muted-foreground/40'
              }`} aria-hidden="true">
                {voice === profile.id && <CheckIcon className="size-3" />}
              </span>
            </label>
          ))}
        </fieldset>
        <p id="voice-help" role="status" className="text-xs text-muted-foreground">
          {locked
            ? `Current voice: ${voice}. End the session to change it.`
            : `${voice} will speak when you connect.`}
        </p>
      </PopoverContent>
    </Popover>
  );
}
