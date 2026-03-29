import { useCallback, useState } from "react";
import { UploadCloud, FileImage, X } from "lucide-react";
import { cn, formatBytes } from "@/lib/utils";

interface FileUploadProps {
  onFileSelect: (file: File | null) => void;
  selectedFile: File | null;
  disabled?: boolean;
}

export function FileUpload({ onFileSelect, selectedFile, disabled }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) return;
    onFileSelect(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  }, [onFileSelect]);

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  };

  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const clearFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    onFileSelect(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
  };

  return (
    <div
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onClick={() => !disabled && !selectedFile && document.getElementById('file-upload')?.click()}
      className={cn(
        "relative w-full h-48 rounded-2xl border-2 border-dashed flex flex-col items-center justify-center transition-all duration-200 overflow-hidden",
        isDragging ? "border-primary bg-primary/5" : "border-white/10 hover:border-white/20 hover:bg-white/5",
        disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
        selectedFile ? "border-solid border-primary/30" : ""
      )}
    >
      <input
        id="file-upload"
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        disabled={disabled}
      />

      {selectedFile && previewUrl ? (
        <div className="absolute inset-0 w-full h-full">
          <img src={previewUrl} alt="Preview" className="w-full h-full object-cover opacity-40 blur-sm" />
          <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-xs p-4">
            <div className="glass-panel p-4 rounded-xl flex items-center gap-4 max-w-[80%]">
              <img src={previewUrl} alt="Thumb" className="w-16 h-16 rounded-lg object-cover border border-white/20" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-white truncate">{selectedFile.name}</p>
                <p className="text-xs text-muted-foreground">{formatBytes(selectedFile.size)}</p>
              </div>
              <button 
                onClick={clearFile}
                disabled={disabled}
                className="p-2 hover:bg-white/10 rounded-full text-muted-foreground hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3 text-center p-6">
          <div className="w-12 h-12 rounded-full bg-white/5 flex items-center justify-center">
            <UploadCloud className="w-6 h-6 text-muted-foreground" />
          </div>
          <div>
            <p className="text-sm font-medium text-white">Drop image tensor here</p>
            <p className="text-xs text-muted-foreground mt-1">or click to browse local files (JPEG, PNG)</p>
          </div>
        </div>
      )}
    </div>
  );
}
