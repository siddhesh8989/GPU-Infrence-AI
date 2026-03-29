import { Layout } from "@/components/layout";
import { useGetMetrics, useGetComparison } from "@workspace/api-client-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, Legend, AreaChart, Area } from "recharts";
import { Activity, Zap, Cpu, Clock, RefreshCw } from "lucide-react";
import { formatMs } from "@/lib/utils";

// Custom tooltip for dark theme charts
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-card/90 backdrop-blur-md border border-white/10 p-3 rounded-lg shadow-xl">
        <p className="text-sm font-medium text-white mb-2">{label}</p>
        {payload.map((entry: any, index: number) => (
          <div key={index} className="flex items-center justify-between gap-4 text-xs font-mono">
            <span style={{ color: entry.color }}>{entry.name}:</span>
            <span className="text-white font-semibold">{entry.value.toFixed(1)}</span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export default function Metrics() {
  const { data: metrics, isLoading: loadingMetrics } = useGetMetrics({
    query: { refetchInterval: 5000 } // Poll every 5s
  });
  
  const { data: comparison, isLoading: loadingComparison } = useGetComparison({
    query: { refetchInterval: 5000 }
  });

  if (loadingMetrics || loadingComparison) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-[60vh]">
          <div className="flex flex-col items-center gap-4">
            <RefreshCw className="w-8 h-8 text-primary animate-spin" />
            <p className="text-muted-foreground animate-pulse">Aggregating telemetry data...</p>
          </div>
        </div>
      </Layout>
    );
  }

  if (!metrics || !comparison) return null;

  return (
    <Layout>
      <div className="space-y-8 pb-12">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-display font-bold text-white">System Analytics</h2>
            <p className="text-muted-foreground mt-1">Aggregated pipeline performance and hardware comparisons</p>
          </div>
        </div>

        {/* Top Stats Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="glass-panel p-5 rounded-2xl relative overflow-hidden">
            <div className="absolute right-0 top-0 p-4 opacity-10"><Activity className="w-16 h-16" /></div>
            <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-1">Total Inferences</p>
            <p className="text-3xl font-bold text-white">{metrics.totalInferences.toLocaleString()}</p>
          </div>
          
          <div className="glass-panel p-5 rounded-2xl border-primary/20 relative overflow-hidden">
            <div className="absolute right-0 top-0 p-4 opacity-10 text-primary"><Zap className="w-16 h-16" /></div>
            <p className="text-xs font-mono text-primary uppercase tracking-wider mb-1">Avg Throughput</p>
            <p className="text-3xl font-bold text-white">{metrics.avgThroughputRps.toFixed(1)} <span className="text-sm font-normal text-muted-foreground">req/s</span></p>
          </div>

          <div className="glass-panel p-5 rounded-2xl relative overflow-hidden">
            <div className="absolute right-0 top-0 p-4 opacity-10"><Clock className="w-16 h-16" /></div>
            <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-1">Avg Latency</p>
            <p className="text-3xl font-bold text-white">{formatMs(metrics.avgLatencyMs)}</p>
          </div>

          <div className="glass-panel p-5 rounded-2xl border-gpu/20 relative overflow-hidden bg-gradient-to-br from-card to-gpu/5">
            <div className="absolute right-0 top-0 p-4 opacity-10 text-gpu"><Zap className="w-16 h-16" /></div>
            <p className="text-xs font-mono text-gpu uppercase tracking-wider mb-1">GPU Speedup</p>
            <p className="text-3xl font-bold text-white">{comparison.speedupFactor.toFixed(1)}x <span className="text-sm font-normal text-muted-foreground">vs CPU</span></p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Latency Comparison Chart */}
          <div className="glass-panel p-6 rounded-2xl">
            <h3 className="text-lg font-display font-medium text-white mb-6">Latency by Architecture (CPU vs GPU)</h3>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparison.latencyComparison} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                  <XAxis dataKey="model" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => `${val}ms`} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                  <Bar dataKey="cpuLatencyMs" name="CPU Latency" fill="hsl(var(--cpu))" radius={[4, 4, 0, 0]} maxBarSize={40} />
                  <Bar dataKey="gpuLatencyMs" name="GPU Latency" fill="hsl(var(--gpu))" radius={[4, 4, 0, 0]} maxBarSize={40} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Throughput Scaling Chart */}
          <div className="glass-panel p-6 rounded-2xl">
            <h3 className="text-lg font-display font-medium text-white mb-6">Throughput Scaling (Batch Size)</h3>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={comparison.throughputComparison} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorGpu" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--gpu))" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="hsl(var(--gpu))" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorCpu" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--cpu))" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="hsl(var(--cpu))" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                  <XAxis dataKey="batchSize" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => `bs=${val}`} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                  <Area type="monotone" dataKey="gpuThroughput" name="GPU (req/s)" stroke="hsl(var(--gpu))" fillOpacity={1} fill="url(#colorGpu)" strokeWidth={2} />
                  <Area type="monotone" dataKey="cpuThroughput" name="CPU (req/s)" stroke="hsl(var(--cpu))" fillOpacity={1} fill="url(#colorCpu)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Model Breakdown Table */}
        <div className="glass-panel p-6 rounded-2xl">
          <h3 className="text-lg font-display font-medium text-white mb-4">Architecture Telemetry Breakdown</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="py-4 px-4 text-xs font-mono text-muted-foreground uppercase">Model Architecture</th>
                  <th className="py-4 px-4 text-xs font-mono text-muted-foreground uppercase text-right">Executions</th>
                  <th className="py-4 px-4 text-xs font-mono text-muted-foreground uppercase text-right">Avg Latency</th>
                  <th className="py-4 px-4 text-xs font-mono text-muted-foreground uppercase text-right">Avg Confidence</th>
                </tr>
              </thead>
              <tbody>
                {metrics.modelBreakdown.map((row, i) => (
                  <tr key={i} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                    <td className="py-4 px-4 text-sm font-medium text-white capitalize">{row.model.replace('_', ' ')}</td>
                    <td className="py-4 px-4 text-sm font-mono text-muted-foreground text-right">{row.count}</td>
                    <td className="py-4 px-4 text-sm font-mono text-white text-right">{formatMs(row.avgLatencyMs)}</td>
                    <td className="py-4 px-4 text-sm font-mono text-primary text-right">{(row.avgConfidence * 100).toFixed(1)}%</td>
                  </tr>
                ))}
                {metrics.modelBreakdown.length === 0 && (
                  <tr>
                    <td colSpan={4} className="py-8 text-center text-muted-foreground text-sm">No telemetry data recorded yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </Layout>
  );
}
