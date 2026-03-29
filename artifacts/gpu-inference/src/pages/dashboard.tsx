import { useState } from "react";
import { Layout } from "@/components/layout";
import { PipelineVisualization } from "@/components/pipeline-visualization";
import { FileUpload } from "@/components/file-upload";
import { useUploadImage, useRunInference, useRunBatchInference, InferenceRequestModel, InferenceRequestComputeMode, InferenceRequestPrecision } from "@workspace/api-client-react";
import { Play, Cpu, Zap, Activity, BarChart3, Clock, HardDrive, Settings2, Layers } from "lucide-react";
import { cn, formatMs } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";

export default function Dashboard() {
  const { toast } = useToast();
  
  // Settings State
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState<InferenceRequestModel>(InferenceRequestModel.resnet50);
  const [computeMode, setComputeMode] = useState<InferenceRequestComputeMode>(InferenceRequestComputeMode.gpu);
  const [batchSize, setBatchSize] = useState<number>(1);
  const [precision, setPrecision] = useState<InferenceRequestPrecision>(InferenceRequestPrecision.fp16);

  // Execution State
  const [pipelineStatus, setPipelineStatus] = useState<'idle' | 'running' | 'success' | 'error'>('idle');

  // Mutations
  const uploadMutation = useUploadImage();
  const inferenceMutation = useRunInference();
  const batchMutation = useRunBatchInference();

  const handleRunInference = async () => {
    if (!file && !inferenceMutation.data) {
      toast({ title: "Error", description: "Please upload an image first", variant: "destructive" });
      return;
    }

    setPipelineStatus('running');
    try {
      let fileId = uploadMutation.data?.fileId;
      
      // If we have a new file that hasn't been uploaded yet
      if (file && (!uploadMutation.data || uploadMutation.variables?.data.file.name !== file.name)) {
        const uploadRes = await uploadMutation.mutateAsync({ data: { file } });
        fileId = uploadRes.fileId;
      }

      if (!fileId) throw new Error("File ID missing");

      await inferenceMutation.mutateAsync({
        data: {
          fileId,
          model,
          computeMode,
          batchSize,
          precision
        }
      });
      setPipelineStatus('success');
      toast({ title: "Inference Complete", description: "Results are ready." });
    } catch (error: any) {
      setPipelineStatus('error');
      toast({ title: "Inference Failed", description: error.message || "An error occurred", variant: "destructive" });
    }
  };

  const handleRunBatch = async () => {
    toast({ title: "Starting Batch Benchmark", description: "Running various batch sizes..." });
    try {
      await batchMutation.mutateAsync({
        data: {
          model,
          computeMode,
          batchSizes: [1, 2, 4, 8, 16, 32],
          precision
        }
      });
      toast({ title: "Benchmark Complete", description: "Check results below." });
    } catch (error: any) {
      toast({ title: "Benchmark Failed", description: error.message || "An error occurred", variant: "destructive" });
    }
  };

  const isRunning = pipelineStatus === 'running' || batchMutation.isPending;
  const result = inferenceMutation.data;
  const batchResult = batchMutation.data;

  return (
    <Layout>
      <div className="space-y-8 pb-12">
        {/* Header Hero */}
        <section className="relative rounded-3xl overflow-hidden border border-white/5 bg-card/30 backdrop-blur-md p-8">
          <div className="absolute inset-0 z-0">
            <img 
              src={`${import.meta.env.BASE_URL}images/ai-grid-bg.png`} 
              alt="AI Grid Background" 
              className="w-full h-full object-cover opacity-20 mix-blend-screen"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent" />
          </div>
          
          <div className="relative z-10">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-mono mb-4">
              <Activity className="w-3.5 h-3.5" /> Core Engine Ready
            </div>
            <h2 className="text-3xl md:text-5xl font-display font-bold text-white mb-4">
              Neural Inference <span className="text-gradient-primary">Pipeline</span>
            </h2>
            <p className="text-muted-foreground max-w-2xl text-sm md:text-base">
              High-performance compute environment for deep learning models. 
              Configure architecture, precision, and hardware acceleration to benchmark throughput and latency.
            </p>
          </div>
        </section>

        {/* Dynamic Pipeline Visualization */}
        <section className="glass-panel rounded-3xl p-6">
          <PipelineVisualization status={pipelineStatus} computeMode={computeMode} />
        </section>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Controls Sidebar */}
          <aside className="lg:col-span-4 space-y-6">
            <div className="glass-panel rounded-2xl p-6 space-y-6">
              <h3 className="font-display text-lg font-semibold text-white flex items-center gap-2 border-b border-white/5 pb-4">
                <Settings2 className="w-5 h-5 text-primary" /> Configuration
              </h3>

              <div className="space-y-5">
                {/* File Upload */}
                <div>
                  <label className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-2 block">Input Tensor (Image)</label>
                  <FileUpload onFileSelect={setFile} selectedFile={file} disabled={isRunning} />
                </div>

                {/* Hardware Toggle */}
                <div>
                  <label className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-2 block">Compute Mode</label>
                  <div className="flex bg-black/40 p-1 rounded-xl border border-white/5">
                    <button
                      onClick={() => setComputeMode('cpu')}
                      disabled={isRunning}
                      className={cn(
                        "flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-all",
                        computeMode === 'cpu' ? "bg-cpu/20 text-cpu shadow-[0_0_15px_rgba(59,130,246,0.2)]" : "text-muted-foreground hover:text-white"
                      )}
                    >
                      <Cpu className="w-4 h-4" /> CPU
                    </button>
                    <button
                      onClick={() => setComputeMode('gpu')}
                      disabled={isRunning}
                      className={cn(
                        "flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-all",
                        computeMode === 'gpu' ? "bg-gpu/20 text-gpu shadow-[0_0_15px_rgba(34,197,94,0.2)]" : "text-muted-foreground hover:text-white"
                      )}
                    >
                      <Zap className="w-4 h-4" /> GPU Accel
                    </button>
                  </div>
                </div>

                {/* Model Select */}
                <div>
                  <label className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-2 block">Architecture</label>
                  <select 
                    value={model}
                    onChange={(e) => setModel(e.target.value as InferenceRequestModel)}
                    disabled={isRunning}
                    className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white text-sm focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all"
                  >
                    <option value="resnet50">ResNet-50 (Vision)</option>
                    <option value="mobilenet">MobileNet V3 (Efficient Vision)</option>
                    <option value="random_forest">Random Forest (Baseline)</option>
                  </select>
                </div>

                {/* Grid for Batch and Precision */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-2 block">Batch Size</label>
                    <select 
                      value={batchSize}
                      onChange={(e) => setBatchSize(Number(e.target.value))}
                      disabled={isRunning}
                      className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white text-sm focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all"
                    >
                      {[1, 2, 4, 8, 16, 32].map(s => (
                        <option key={s} value={s}>{s}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-2 block">Precision</label>
                    <select 
                      value={precision}
                      onChange={(e) => setPrecision(e.target.value as InferenceRequestPrecision)}
                      disabled={isRunning}
                      className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white text-sm focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all"
                    >
                      <option value="fp32">FP32 (32-bit)</option>
                      <option value="fp16">FP16 (16-bit)</option>
                    </select>
                  </div>
                </div>
              </div>

              <div className="pt-4 space-y-3">
                <button
                  onClick={handleRunInference}
                  disabled={isRunning || (!file && !result)}
                  className={cn(
                    "w-full flex items-center justify-center gap-2 py-4 rounded-xl font-bold text-lg transition-all duration-300 relative overflow-hidden group",
                    isRunning ? "bg-white/10 text-white/50 cursor-wait" : 
                    computeMode === 'gpu' 
                      ? "bg-gpu text-gpu-foreground hover:shadow-[0_0_30px_rgba(34,197,94,0.4)] hover:scale-[1.02]" 
                      : "bg-primary text-primary-foreground hover:shadow-[0_0_30px_rgba(59,130,246,0.4)] hover:scale-[1.02]"
                  )}
                >
                  {isRunning ? (
                    <div className="flex items-center gap-2">
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Processing...
                    </div>
                  ) : (
                    <>
                      <Play className="w-5 h-5 fill-current" /> Execute Inference
                    </>
                  )}
                  {/* Button shine effect */}
                  <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/20 to-transparent group-hover:animate-[shimmer_1.5s_infinite]" />
                </button>

                <button
                  onClick={handleRunBatch}
                  disabled={isRunning}
                  className="w-full flex items-center justify-center gap-2 py-3 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 text-white text-sm font-medium transition-colors"
                >
                  <BarChart3 className="w-4 h-4" /> Run Batch Benchmark
                </button>
              </div>
            </div>
          </aside>

          {/* Results Area */}
          <main className="lg:col-span-8 space-y-6">
            {result ? (
              <div className="space-y-6 animate-in slide-in-from-bottom-4 duration-500">
                {/* Main Prediction Card */}
                <div className="glass-panel rounded-2xl p-8 relative overflow-hidden">
                  <div className="absolute right-0 top-0 w-64 h-64 bg-primary/10 rounded-full blur-[80px] -translate-y-1/2 translate-x-1/4 pointer-events-none" />
                  
                  <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 relative z-10">
                    <div>
                      <p className="text-sm font-mono text-muted-foreground uppercase tracking-wider mb-1">Top Prediction</p>
                      <h3 className="text-4xl md:text-5xl font-display font-bold text-white capitalize">
                        {result.prediction}
                      </h3>
                      <div className="mt-4 flex items-center gap-3">
                        <div className="h-2 w-48 bg-black/50 rounded-full overflow-hidden">
                          <div 
                            className={cn("h-full rounded-full", result.confidence > 0.8 ? "bg-gpu" : result.confidence > 0.5 ? "bg-amber-400" : "bg-destructive")}
                            style={{ width: `${result.confidence * 100}%` }}
                          />
                        </div>
                        <span className="font-mono text-sm font-medium text-white">
                          {(result.confidence * 100).toFixed(2)}%
                        </span>
                      </div>
                    </div>
                    
                    <div className="flex flex-wrap gap-2">
                      <span className="px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs font-mono text-muted-foreground flex items-center gap-1.5">
                        <Clock className="w-3 h-3" /> {formatMs(result.timings.totalMs)} total
                      </span>
                      <span className="px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs font-mono text-muted-foreground flex items-center gap-1.5">
                        <Zap className="w-3 h-3" /> {result.timings.throughputRps.toFixed(1)} req/s
                      </span>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Performance Breakdown */}
                  <div className="glass-panel rounded-2xl p-6">
                    <h4 className="font-display text-lg font-medium text-white mb-4 flex items-center gap-2">
                      <Activity className="w-4 h-4 text-primary" /> Execution Telemetry
                    </h4>
                    <div className="space-y-4">
                      {result.pipelineStages.map((stage) => (
                        <div key={stage.name} className="flex flex-col gap-1.5">
                          <div className="flex justify-between text-sm">
                            <span className="text-muted-foreground">{stage.name}</span>
                            <span className="font-mono text-white">{formatMs(stage.durationMs)}</span>
                          </div>
                          <div className="h-1.5 w-full bg-black/40 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-primary/60 rounded-full"
                              style={{ width: `${Math.max(2, (stage.durationMs / result.timings.totalMs) * 100)}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                    
                    <div className="mt-6 pt-6 border-t border-white/5">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-black/20 p-3 rounded-xl border border-white/5">
                          <p className="text-xs text-muted-foreground font-mono mb-1 flex items-center gap-1"><HardDrive className="w-3 h-3" /> Peak VRAM</p>
                          <p className="text-lg font-semibold text-white">{result.memoryUsage.gpuSimulatedMb.toFixed(0)} MB</p>
                        </div>
                        <div className="bg-black/20 p-3 rounded-xl border border-white/5">
                          <p className="text-xs text-muted-foreground font-mono mb-1 flex items-center gap-1"><HardDrive className="w-3 h-3" /> Heap Used</p>
                          <p className="text-lg font-semibold text-white">{result.memoryUsage.heapUsedMb.toFixed(0)} MB</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Top 5 Classes */}
                  <div className="glass-panel rounded-2xl p-6">
                    <h4 className="font-display text-lg font-medium text-white mb-4 flex items-center gap-2">
                      <Layers className="w-4 h-4 text-primary" /> Class Probabilities
                    </h4>
                    <div className="space-y-3">
                      {result.topPredictions.map((pred, idx) => (
                        <div key={idx} className="flex items-center justify-between p-3 rounded-xl bg-black/20 border border-white/5 hover:bg-white/5 transition-colors">
                          <div className="flex items-center gap-3">
                            <span className="text-xs font-mono text-muted-foreground w-4">{pred.rank}.</span>
                            <span className="text-sm font-medium text-white capitalize">{pred.label}</span>
                          </div>
                          <span className="text-sm font-mono text-muted-foreground">
                            {(pred.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ) : batchResult ? (
              <div className="space-y-6 animate-in slide-in-from-bottom-4 duration-500">
                 <div className="glass-panel rounded-2xl p-8">
                    <h3 className="text-2xl font-display font-bold text-white mb-2">Batch Benchmark Results</h3>
                    <p className="text-muted-foreground text-sm mb-6">Model: {batchResult.model} | Mode: {batchResult.computeMode} | Precision: {batchResult.precision}</p>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                      <div className="bg-primary/10 border border-primary/20 p-4 rounded-xl">
                        <p className="text-xs text-primary font-mono mb-1">Peak Throughput</p>
                        <p className="text-2xl font-bold text-white">{batchResult.summary.peakThroughput.toFixed(0)} <span className="text-sm font-normal text-muted-foreground">req/s</span></p>
                      </div>
                      <div className="bg-white/5 border border-white/10 p-4 rounded-xl">
                        <p className="text-xs text-muted-foreground font-mono mb-1">Best Batch Size</p>
                        <p className="text-2xl font-bold text-white">{batchResult.summary.bestBatchSize}</p>
                      </div>
                      <div className="bg-white/5 border border-white/10 p-4 rounded-xl">
                        <p className="text-xs text-muted-foreground font-mono mb-1">Min Latency</p>
                        <p className="text-2xl font-bold text-white">{batchResult.summary.minLatency.toFixed(1)} <span className="text-sm font-normal text-muted-foreground">ms</span></p>
                      </div>
                      <div className="bg-white/5 border border-white/10 p-4 rounded-xl">
                        <p className="text-xs text-muted-foreground font-mono mb-1">Avg VRAM</p>
                        <p className="text-2xl font-bold text-white">{batchResult.summary.avgMemoryMb.toFixed(0)} <span className="text-sm font-normal text-muted-foreground">MB</span></p>
                      </div>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-left border-collapse">
                        <thead>
                          <tr className="border-b border-white/10">
                            <th className="py-3 px-4 text-xs font-mono text-muted-foreground uppercase">Batch Size</th>
                            <th className="py-3 px-4 text-xs font-mono text-muted-foreground uppercase">Latency (ms)</th>
                            <th className="py-3 px-4 text-xs font-mono text-muted-foreground uppercase">Throughput (req/s)</th>
                            <th className="py-3 px-4 text-xs font-mono text-muted-foreground uppercase">Memory (MB)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {batchResult.results.map((r, i) => (
                            <tr key={i} className="border-b border-white/5 hover:bg-white/5">
                              <td className="py-3 px-4 text-sm font-bold text-white">{r.batchSize}</td>
                              <td className="py-3 px-4 text-sm font-mono text-muted-foreground">{r.latencyMs.toFixed(2)}</td>
                              <td className="py-3 px-4 text-sm font-mono text-primary font-medium">{r.throughputRps.toFixed(1)}</td>
                              <td className="py-3 px-4 text-sm font-mono text-muted-foreground">{r.memoryMb.toFixed(0)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                 </div>
              </div>
            ) : (
              <div className="h-full min-h-[400px] glass-panel rounded-2xl flex flex-col items-center justify-center p-12 text-center border-dashed">
                <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mb-4">
                  <Activity className="w-8 h-8 text-muted-foreground" />
                </div>
                <h3 className="text-xl font-display font-medium text-white mb-2">Awaiting Inference Task</h3>
                <p className="text-muted-foreground max-w-sm">
                  Upload an image tensor and configure the pipeline parameters to execute a forward pass.
                </p>
              </div>
            )}
          </main>
        </div>
      </div>
    </Layout>
  );
}
