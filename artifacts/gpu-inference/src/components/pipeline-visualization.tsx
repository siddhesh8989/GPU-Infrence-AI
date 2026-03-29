import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { UploadCloud, Settings2, Layers, Cpu, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface PipelineVisualizationProps {
  status: 'idle' | 'running' | 'success' | 'error';
  computeMode: 'cpu' | 'gpu';
}

const STAGES = [
  { id: 'input', label: 'Input Buffer', icon: UploadCloud },
  { id: 'preproc', label: 'Pre-process', icon: Settings2 },
  { id: 'batch', label: 'Batching', icon: Layers },
  { id: 'engine', label: 'Inference Engine', icon: Cpu },
  { id: 'output', label: 'Output Tensor', icon: CheckCircle2 },
];

export function PipelineVisualization({ status, computeMode }: PipelineVisualizationProps) {
  const [activeStageIndex, setActiveStageIndex] = useState(-1);

  // Simulate stage progression
  useEffect(() => {
    if (status === 'idle') {
      setActiveStageIndex(-1);
    } else if (status === 'success' || status === 'error') {
      setActiveStageIndex(4); // Snap to end
    } else if (status === 'running') {
      setActiveStageIndex(0);
      let current = 0;
      const interval = setInterval(() => {
        current = (current + 1) % 5;
        // Don't auto-complete the last stage until status changes
        if (current < 4) {
          setActiveStageIndex(current);
        }
      }, 400); // 400ms per stage simulation
      
      return () => clearInterval(interval);
    }
  }, [status]);

  const themeColor = computeMode === 'gpu' ? 'var(--gpu)' : 'var(--cpu)';
  const glowClass = computeMode === 'gpu' ? 'shadow-[0_0_20px_var(--gpu)]' : 'shadow-[0_0_20px_var(--cpu)]';

  return (
    <div className="w-full py-8 overflow-hidden">
      <div className="flex items-center justify-between relative max-w-4xl mx-auto px-4">
        
        {/* Connection Lines (Background) */}
        <div className="absolute left-[10%] right-[10%] top-1/2 -translate-y-1/2 h-1 bg-white/5 rounded-full z-0" />
        
        {/* Animated Active Line */}
        <div className="absolute left-[10%] right-[10%] top-1/2 -translate-y-1/2 h-1 z-0 flex overflow-hidden rounded-full">
          <motion.div 
            className="h-full"
            style={{ backgroundColor: `hsl(${themeColor})` }}
            initial={{ width: "0%" }}
            animate={{ 
              width: activeStageIndex >= 0 ? `${(activeStageIndex / 4) * 100}%` : "0%"
            }}
            transition={{ duration: 0.4, ease: "easeInOut" }}
          />
        </div>

        {/* Nodes */}
        {STAGES.map((stage, i) => {
          const Icon = stage.icon;
          const isPast = activeStageIndex > i || status === 'success';
          const isActive = activeStageIndex === i && status === 'running';
          const isError = status === 'error' && i === 4;

          return (
            <div key={stage.id} className="relative z-10 flex flex-col items-center gap-3">
              <motion.div
                className={cn(
                  "w-12 h-12 rounded-xl flex items-center justify-center border-2 transition-colors duration-300",
                  isActive ? `bg-black border-[hsl(${themeColor})] ${glowClass}` :
                  isPast ? `bg-[hsl(${themeColor})/0.1] border-[hsl(${themeColor})/0.5] text-[hsl(${themeColor})]` :
                  isError ? "bg-destructive/10 border-destructive text-destructive" :
                  "bg-card border-white/10 text-muted-foreground"
                )}
                animate={{
                  scale: isActive ? [1, 1.1, 1] : 1,
                }}
                transition={{ repeat: isActive ? Infinity : 0, duration: 1.5 }}
              >
                <Icon className={cn(
                  "w-5 h-5",
                  isActive ? `text-[hsl(${themeColor})]` : "",
                  isPast && !isActive ? `text-[hsl(${themeColor})]` : ""
                )} />
              </motion.div>
              <span className={cn(
                "text-xs font-mono font-medium absolute -bottom-6 whitespace-nowrap",
                isActive ? "text-white" : "text-muted-foreground"
              )}>
                {stage.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
