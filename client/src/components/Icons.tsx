import type { SVGProps } from "react";

/** Gold monogram used in the sidebar, login page and splash screen. */
export function LogoMark({ size = 40 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" aria-hidden="true" className="logo-mark">
      <defs>
        <linearGradient id="logo-gold" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#f6dfa8" />
          <stop offset="55%" stopColor="#d4af6a" />
          <stop offset="100%" stopColor="#9c7535" />
        </linearGradient>
      </defs>
      <rect x="1.5" y="1.5" width="45" height="45" rx="13" fill="#0d0f16" stroke="url(#logo-gold)" strokeWidth="1.5" />
      <path d="M11 32 L19 23 L25 28 L36 15" fill="none" stroke="url(#logo-gold)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M30 15 H36 V21" fill="none" stroke="url(#logo-gold)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
      <rect x="11" y="35.5" width="26" height="1.6" rx="0.8" fill="url(#logo-gold)" opacity="0.55" />
    </svg>
  );
}

type IconProps = SVGProps<SVGSVGElement>;

function Icon({ children, ...props }: IconProps) {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  );
}

export const HomeIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M3 10.5 12 3l9 7.5" />
    <path d="M5 9.5V21h14V9.5" />
    <path d="M10 21v-6h4v6" />
  </Icon>
);

export const SimulationsIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M4 19V5" />
    <path d="M4 19h16" />
    <path d="M7 15l4-5 3 3 5-7" />
  </Icon>
);

export const ChatIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12Z" />
    <path d="M9 11h.01M12 11h.01M15 11h.01" strokeWidth="2.4" />
  </Icon>
);

export const StrategyIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 3 4 7v5c0 4.5 3.4 8.3 8 9 4.6-.7 8-4.5 8-9V7Z" />
    <path d="m9 12 2 2 4-4" />
  </Icon>
);

export const LiveIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M7 4v3M7 17v3M17 4v5M17 19v1" />
    <rect x="5" y="7" width="4" height="10" rx="1" />
    <rect x="15" y="9" width="4" height="10" rx="1" />
  </Icon>
);

export const PortfolioIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 3a9 9 0 1 0 9 9h-9Z" />
    <path d="M15 3.5A9 9 0 0 1 20.5 9H15Z" />
  </Icon>
);

export const BuilderIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M4 7h10M18 7h2M4 17h4M12 17h8" />
    <circle cx="16" cy="7" r="2" />
    <circle cx="10" cy="17" r="2" />
  </Icon>
);

export const LabIcon = (p: IconProps) => (
  <Icon {...p}>
    <rect x="3.5" y="12" width="4" height="8" rx="1" />
    <rect x="10" y="7" width="4" height="13" rx="1" />
    <rect x="16.5" y="4" width="4" height="16" rx="1" />
  </Icon>
);

export const TrophyIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M8 4h8v5a4 4 0 0 1-8 0Z" />
    <path d="M8 6H5a3 3 0 0 0 3 4M16 6h3a3 3 0 0 1-3 4M12 13v4M8.5 20h7M10 17h4" />
  </Icon>
);

export const SipIcon = (p: IconProps) => (
  <Icon {...p}>
    <rect x="4" y="5" width="16" height="15" rx="2" />
    <path d="M4 10h16M9 3v4M15 3v4M8.5 14.5l2 2 4-4" />
  </Icon>
);

export const OptionsIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M3 17h7l8-10h3" />
    <path d="M3 7h4M14 17h7" />
  </Icon>
);

export const LearnIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M2.5 9 12 4.5 21.5 9 12 13.5Z" />
    <path d="M6.5 11v4.5c1.5 1.4 3.4 2 5.5 2s4-.6 5.5-2V11M21.5 9v5" />
  </Icon>
);

export const LayersIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 3 3 8l9 5 9-5Z" />
    <path d="m3 12.5 9 5 9-5M3 17l9 5 9-5" />
  </Icon>
);

export const VisitorsIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="9" cy="8" r="3.5" />
    <path d="M2.5 20c.8-3.6 3.4-5.5 6.5-5.5s5.7 1.9 6.5 5.5" />
    <path d="M16 4.6a3.5 3.5 0 0 1 0 6.8M18.5 14.8c1.6.8 2.6 2.5 3 5.2" />
  </Icon>
);

export const LogoutIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M15 4h3a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-3" />
    <path d="M10 16l-4-4 4-4M6 12h10" />
  </Icon>
);

export const ShieldIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 3 4 7v5c0 4.5 3.4 8.3 8 9 4.6-.7 8-4.5 8-9V7Z" />
  </Icon>
);

export const KeyIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="8" cy="15" r="4" />
    <path d="m11 12 9-9M17 6l3 3M15 8l2 2" />
  </Icon>
);

export const LockIcon = (p: IconProps) => (
  <Icon {...p}>
    <rect x="4.5" y="10.5" width="15" height="10" rx="2" />
    <path d="M8 10.5V7a4 4 0 0 1 8 0v3.5" />
  </Icon>
);
