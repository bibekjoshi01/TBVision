"use client";

import { useState } from "react";
import { UploadCloud, Pencil, Trash2 } from "lucide-react";
import { Image } from "@heroui/image";
import { Button } from "@heroui/button";

interface RadiographAcquisitionProps {
  previewUrl: string | null;
  handleImageUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  removeImage: (e?: any) => void;
}

export function RadiographAcquisition({ previewUrl, handleImageUpload, removeImage }: RadiographAcquisitionProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.type.startsWith("image/")) {
        // Simulate a change event to reuse the existing handleImageUpload
        const syntheticEvent = {
          target: { files },
        } as unknown as React.ChangeEvent<HTMLInputElement>;
        handleImageUpload(syntheticEvent);
      }
    }
  };

  return (
    <div className="space-y-4">
      <p className="text-xs font-bold text-default-600 uppercase tracking-widest flex items-center gap-2">
        <UploadCloud size={14} /> 1. Radiograph Acquisition
      </p>
      <div
        className={`relative group overflow-hidden rounded-xl border-2 border-dashed transition-all aspect-[2/1] md:aspect-[3/1] flex flex-col items-center justify-center text-center
          ${isDragging ? "border-primary bg-primary/5" : "border-default-200 hover:border-primary bg-default-50"}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {previewUrl ? (
          <div className="relative w-full h-full z-10">
            <Image
              src={previewUrl}
              alt="Preview"
              className="w-full h-full object-cover rounded-none"
              removeWrapper
            />
            <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-all duration-300 flex items-center justify-center gap-8 z-30">
               <div className="flex flex-col items-center gap-3">
                 <Button
                   isIconOnly
                   radius="full"
                   variant="solid"
                   className="bg-white text-primary w-14 h-14 min-w-0"
                   onPress={() => document.getElementById('image-input')?.click()}
                 >
                   <Pencil size={24} />
                 </Button>
                 <span className="text-white text-[11px] font-black uppercase tracking-widest">Change Image</span>
               </div>
               <div className="flex flex-col items-center gap-3">
                 <Button
                   isIconOnly
                   radius="full"
                   variant="solid"
                   className="bg-danger text-white w-14 h-14 min-w-0"
                   onPress={removeImage}
                 >
                   <Trash2 size={24} />
                 </Button>
                 <span className="text-white text-[11px] font-black uppercase tracking-widest">Clear File</span>
               </div>
            </div>
          </div>
        ) : (
          <div className="p-4 space-y-4 pointer-events-none">
            <div className="w-20 h-20 rounded-full bg-default-100 flex items-center justify-center mx-auto mb-2 border-2 border-dashed border-default-200">
              <UploadCloud size={36} className={`transition-colors ${isDragging ? "text-primary" : "text-default-400"}`} />
            </div>
            <p className="text-base font-bold text-default-600">
              {isDragging ? "Release to upload" : "Drop radiograph or click to browse"}
            </p>
            <p className="text-sm text-default-400">Accepted: JPG, JPEG, PNG</p>
          </div>
        )}
        <input
          id="image-input"
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          className={`absolute inset-0 opacity-0 cursor-pointer ${previewUrl ? 'z-0' : 'z-20'}`}
        />
      </div>
    </div>
  );
}
