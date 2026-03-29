import { Layout } from "@/components/layout";
import { useGetHistory, useClearMetrics } from "@workspace/api-client-react";
import { format } from "date-fns";
import { formatMs, cn } from "@/lib/utils";
import { Trash2, Cpu, Zap, Search } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useQueryClient } from "@tanstack/react-query";

export default function History() {
  const { data, isLoading } = useGetHistory();
  const clearMutation = useClearMetrics();
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const handleClear = async () => {
    if (confirm("Are you sure you want to clear all telemetry history?")) {
      try {
        await clearMutation.mutateAsync();
        queryClient.invalidateQueries({ queryKey: ['/api/metrics/history'] });
        queryClient.invalidateQueries({ queryKey: ['/api/metrics'] });
        queryClient.invalidateQueries({ queryKey: ['/api/metrics/compare'] });
        toast({ title: "History Cleared", description: "All telemetry data has been purged." });
      } catch (e) {
        toast({ title: "Error", description: "Failed to clear history", variant: "destructive" });
      }
    }
  };

  return (
    <Layout>
      <div className="space-y-6 pb-12">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-3xl font-display font-bold text-white">Execution Log</h2>
            <p className="text-muted-foreground mt-1">Record of all forward passes through the pipeline</p>
          </div>
          <button
            onClick={handleClear}
            disabled={clearMutation.isPending || !data?.items.length}
            className="flex items-center gap-2 px-4 py-2 rounded-xl border border-destructive/30 text-destructive hover:bg-destructive/10 transition-colors text-sm font-medium disabled:opacity-50"
          >
            <Trash2 className="w-4 h-4" /> Purge Logs
          </button>
        </div>

        <div className="glass-panel rounded-2xl overflow-hidden border border-white/10">
          <div className="p-4 border-b border-white/5 flex items-center gap-3 bg-white/5">
            <Search className="w-4 h-4 text-muted-foreground" />
            <input 
              type="text" 
              placeholder="Filter by architecture or prediction..." 
              className="bg-transparent border-none text-sm text-white focus:outline-none w-full placeholder:text-muted-foreground/50"
              disabled // visual only for this mockup unless we implement client side filter
            />
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-black/20">
                  <th className="py-3 px-6 text-xs font-mono text-muted-foreground uppercase tracking-wider">Timestamp</th>
                  <th className="py-3 px-6 text-xs font-mono text-muted-foreground uppercase tracking-wider">Architecture</th>
                  <th className="py-3 px-6 text-xs font-mono text-muted-foreground uppercase tracking-wider">Mode</th>
                  <th className="py-3 px-6 text-xs font-mono text-muted-foreground uppercase tracking-wider">Batch</th>
                  <th className="py-3 px-6 text-xs font-mono text-muted-foreground uppercase tracking-wider">Latency</th>
                  <th className="py-3 px-6 text-xs font-mono text-muted-foreground uppercase tracking-wider">Top Prediction</th>
                  <th className="py-3 px-6 text-xs font-mono text-muted-foreground uppercase tracking-wider text-right">Conf.</th>
                </tr>
              </thead>
              <tbody>
                {isLoading ? (
                  Array.from({ length: 5 }).map((_, i) => (
                    <tr key={i} className="border-t border-white/5">
                      <td colSpan={7} className="py-4 px-6">
                        <div className="h-5 bg-white/5 rounded animate-pulse w-full"></div>
                      </td>
                    </tr>
                  ))
                ) : data?.items.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="py-12 text-center text-muted-foreground">
                      <History className="w-8 h-8 mx-auto mb-3 opacity-20" />
                      <p>No execution history found.</p>
                    </td>
                  </tr>
                ) : (
                  data?.items.map((item) => (
                    <tr key={item.id} className="border-t border-white/5 hover:bg-white/5 transition-colors group">
                      <td className="py-3 px-6 text-sm font-mono text-muted-foreground whitespace-nowrap">
                        {format(new Date(item.timestamp), 'HH:mm:ss.SSS')}
                      </td>
                      <td className="py-3 px-6 text-sm font-medium text-white capitalize whitespace-nowrap">
                        {item.model.replace('_', ' ')}
                      </td>
                      <td className="py-3 px-6 text-sm">
                        <span className={cn(
                          "inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-mono font-medium",
                          item.computeMode === 'gpu' ? "bg-gpu/20 text-gpu" : "bg-cpu/20 text-cpu"
                        )}>
                          {item.computeMode === 'gpu' ? <Zap className="w-3 h-3" /> : <Cpu className="w-3 h-3" />}
                          {item.computeMode.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-3 px-6 text-sm font-mono text-muted-foreground">
                        {item.batchSize}
                      </td>
                      <td className="py-3 px-6 text-sm font-mono text-white">
                        {formatMs(item.latencyMs)}
                      </td>
                      <td className="py-3 px-6 text-sm font-medium text-white capitalize truncate max-w-[200px]">
                        {item.prediction}
                      </td>
                      <td className="py-3 px-6 text-sm font-mono text-right">
                        <span className={cn(
                          item.confidence > 0.8 ? "text-gpu" : item.confidence > 0.5 ? "text-amber-400" : "text-destructive"
                        )}>
                          {(item.confidence * 100).toFixed(1)}%
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          {data && data.items.length > 0 && (
             <div className="p-4 border-t border-white/5 bg-black/20 flex justify-between items-center text-xs font-mono text-muted-foreground">
               <span>Showing {data.items.length} records</span>
               <span>Total historical records: {data.total}</span>
             </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
