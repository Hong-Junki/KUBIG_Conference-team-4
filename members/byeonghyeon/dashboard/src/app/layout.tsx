import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Conflict Temperature Map',
  description: 'ACLED + GDELT + Economic Indicators 기반 무력충돌 조기경보 대시보드',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body className="antialiased">{children}</body>
    </html>
  );
}
