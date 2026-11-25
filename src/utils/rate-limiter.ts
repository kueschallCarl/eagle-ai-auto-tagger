export class RateLimiter {
  private tokens: number;
  private lastRefillTime: number;
  private readonly maxTokens: number;
  private readonly refillRatePerSecond: number;

  constructor(maxRequestsPerMinute: number) {
    this.maxTokens = maxRequestsPerMinute;
    this.tokens = maxRequestsPerMinute;
    this.refillRatePerSecond = maxRequestsPerMinute / 60;
    this.lastRefillTime = Date.now();
  }

  async waitForToken(): Promise<void> {
    this.refill();

    while (this.tokens < 1) {
      const waitMs = Math.ceil((1 - this.tokens) / this.refillRatePerSecond * 1000);
      await new Promise(resolve => setTimeout(resolve, Math.min(waitMs, 1000)));
      this.refill();
    }

    this.tokens -= 1;
  }

  private refill(): void {
    const now = Date.now();
    const elapsed = (now - this.lastRefillTime) / 1000;
    this.tokens = Math.min(this.maxTokens, this.tokens + elapsed * this.refillRatePerSecond);
    this.lastRefillTime = now;
  }
}

const limiters: Record<string, RateLimiter> = {};

export function getRateLimiter(provider: string): RateLimiter {
  if (!limiters[provider]) {
    const limits: Record<string, number> = {
      openrouter: 20,
      groq: 30,
    };
    limiters[provider] = new RateLimiter(limits[provider] || 20);
  }
  return limiters[provider];
}
