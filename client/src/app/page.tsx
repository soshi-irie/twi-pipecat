'use client';

import { useState } from 'react';

import { Console } from '@/components/pipecat/console/console';
import {
  AVAILABLE_TRANSPORTS,
  DEFAULT_TRANSPORT,
  PROJECT_NAME,
  TRANSPORT_FACTORIES,
  transportPropsForSpeech,
} from '@/config';
import type { TransportType } from '@/config';
import { DEFAULT_VOICE, type VoiceProfile } from '@/lib/voice-profiles';
import { DEFAULT_DIALECT, type TwiDialect } from '@/lib/twi-dialects';

import { DialectSelect } from './components/DialectSelect';
import { TransportSelect } from './components/TransportSelect';
import { VoiceSelect } from './components/VoiceSelect';

/**
 * The Pipecat UI console: connect flow, transcript, metrics, device and
 * session info, and a live event stream. It is composed from the components
 * under src/components/pipecat, which are yours to build a custom UI from;
 * the README shows the minimal composition.
 */
export default function Home() {
  const [voice, setVoice] = useState<VoiceProfile>(DEFAULT_VOICE);
  const [dialect, setDialect] = useState<TwiDialect>(DEFAULT_DIALECT);
  const [transportType, setTransportType] =
    useState<TransportType>(DEFAULT_TRANSPORT);

  return (
    <div className="h-dvh">
      <Console
        // The console reads its transport factory once; remount to switch.
        key={transportType}
        transportType={transportType}
        transportFactory={TRANSPORT_FACTORIES[transportType]}
        {...transportPropsForSpeech(transportType, voice, dialect)}
        titleText={PROJECT_NAME}
        headerSlot={(app) => (
          <>
            <VoiceSelect voice={voice} onVoiceChange={setVoice} isConnecting={app.isConnecting} />
            <DialectSelect dialect={dialect} onDialectChange={setDialect} isConnecting={app.isConnecting} />
            {AVAILABLE_TRANSPORTS.length > 1 && (
              <TransportSelect
                transportType={transportType}
                onTransportChange={setTransportType}
                availableTransports={AVAILABLE_TRANSPORTS}
              />
            )}
          </>
        )}
      />
    </div>
  );
}
