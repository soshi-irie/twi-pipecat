export const TWI_DIALECTS = [
  { id: 'asante', label: 'Asante Twi', shortLabel: 'Asante' },
  { id: 'akuapem', label: 'Akuapem Twi', shortLabel: 'Akuapem' },
] as const;

export type TwiDialect = (typeof TWI_DIALECTS)[number]['id'];
export const DEFAULT_DIALECT: TwiDialect = 'asante';

export function isTwiDialect(value: unknown): value is TwiDialect {
  return TWI_DIALECTS.some((dialect) => dialect.id === value);
}
