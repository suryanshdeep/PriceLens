import { ImagePlus, X } from "lucide-react";

type ImageUploadProps = {
  file: File | null;
  onChange: (file: File | null) => void;
};

export default function ImageUpload({ file, onChange }: ImageUploadProps) {
  return (
    <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 p-4">
      <label className="flex cursor-pointer flex-col items-center justify-center gap-3 text-center">
        <ImagePlus className="h-8 w-8 text-steel" aria-hidden="true" />
        <div>
          <p className="text-sm font-medium text-slate-800">
            {file ? file.name : "Attach product image"}
          </p>
          <p className="text-xs text-slate-500">Optional for v1, accepted by the backend contract</p>
        </div>
        <input
          className="sr-only"
          type="file"
          accept="image/*"
          onChange={(event) => onChange(event.target.files?.[0] ?? null)}
        />
      </label>
      {file ? (
        <button
          type="button"
          className="mt-3 inline-flex items-center gap-2 rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-700 hover:bg-white"
          onClick={() => onChange(null)}
        >
          <X className="h-4 w-4" aria-hidden="true" />
          Remove
        </button>
      ) : null}
    </div>
  );
}
