export const VOICE_PROFILES = [
  { id: 'Sulafat', tone: 'Warm', gender: 'Female', description: 'A welcoming, reassuring presence.' },
  { id: 'Callirrhoe', tone: 'Easy-going', gender: 'Female', description: 'Relaxed and naturally conversational.' },
  { id: 'Achird', tone: 'Friendly', gender: 'Male', description: 'Approachable and easy to talk to.' },
] as const;

export type VoiceProfile = (typeof VOICE_PROFILES)[number]['id'];
export const DEFAULT_VOICE: VoiceProfile = 'Callirrhoe';

export function isVoiceProfile(value: unknown): value is VoiceProfile {
  return VOICE_PROFILES.some((profile) => profile.id === value);
}
