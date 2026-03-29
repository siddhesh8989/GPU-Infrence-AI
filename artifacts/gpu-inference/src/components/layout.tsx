import { ReactNode } from "react";
import { Link, useLocation } from "wouter";
import { Activity, Cpu, Database, History, LayoutDashboard, Settings } from "lucide-react";
import { cn } from "@/lib/utils";
import { useGetMetrics } from "@workspace/api-client-react";

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const [location] = useLocation();
  const { data: metrics } = useGetMetrics();

  const navItems = [
    { href: "/", label: "Pipeline", icon: LayoutDashboard },
    { href: "/metrics", label: "Analytics", icon: Activity },
    { href: "/history", label: "History", icon: History },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-background relative overflow-hidden">
      {/* Abstract background glow */}
      <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-primary/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-0 right-1/4 w-[600px] h-[600px] bg-accent/5 rounded-full blur-[150px] pointer-events-none" />

      {/* Top Navbar */}
      <header className="sticky top-0 z-50 glass-panel border-b border-white/5">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-accent flex items-center justify-center shadow-lg shadow-primary/20">
              <Cpu className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-display font-bold text-gradient tracking-wide">
              Tensor<span className="text-primary font-light">Flow</span>
            </h1>
          </div>

          <nav className="flex items-center gap-1 bg-white/5 p-1 rounded-xl border border-white/5">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300",
                    isActive 
                      ? "bg-white/10 text-white shadow-sm" 
                      : "text-muted-foreground hover:text-white hover:bg-white/5"
                  )}
                >
                  <Icon className="w-4 h-4" />
                  {item.label}
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground bg-black/40 px-3 py-1.5 rounded-full border border-white/5">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-gpu opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-gpu"></span>
              </span>
              SYSTEM ONLINE
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 lg:p-8 relative z-10">
        {children}
      </main>

      {/* System Info Footer */}
      <footer className="border-t border-white/5 bg-black/20 backdrop-blur-sm mt-auto relative z-20">
        <div className="max-w-7xl mx-auto px-4 py-4 flex flex-col md:flex-row items-center justify-between gap-4 text-xs font-mono text-muted-foreground">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <Database className="w-3.5 h-3.5" />
              <span>Mem: {metrics?.systemInfo ? `${Math.round(metrics.systemInfo.totalMemoryMb / 1024)}GB` : '---'}</span>
            </div>
            <div className="flex items-center gap-2">
              <Cpu className="w-3.5 h-3.5" />
              <span>Cores: {metrics?.systemInfo?.cpuCores || '---'}</span>
            </div>
            <div className="flex items-center gap-2">
              <Settings className="w-3.5 h-3.5" />
              <span>Node: {metrics?.systemInfo?.nodeVersion || '---'}</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            GPU Accelerators: 
            <span className={cn(
              "font-bold px-2 py-0.5 rounded",
              metrics?.systemInfo?.gpuAvailable 
                ? "bg-gpu/20 text-gpu" 
                : "bg-destructive/20 text-destructive"
            )}>
              {metrics?.systemInfo?.gpuAvailable ? 'DETECTED' : 'UNAVAILABLE'}
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
