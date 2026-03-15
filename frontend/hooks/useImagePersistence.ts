import { useState, useCallback } from "react";
import { saveFile, getFile, clearStorage } from "@/lib/db";

const IMAGE_KEY = "latest_upload";

export interface UseImagePersistenceReturn {
  selectedFile: File | null;
  previewUrl: string | null;
  handleImageUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  removeImage: () => void;
  restoreImage: () => Promise<void>;
  setPreviewFromFile: (file: File) => void;
}

export function useImagePersistence(): UseImagePersistenceReturn {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const setPreviewFromFile = useCallback((file: File) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  }, []);

  const handleImageUpload = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      setPreviewFromFile(file);
      saveFile(IMAGE_KEY, file).catch(console.error);
    },
    [setPreviewFromFile]
  );

  const removeImage = useCallback(() => {
    setPreviewUrl(null);
    setSelectedFile(null);
    clearStorage().catch(console.error);
  }, []);

  const restoreImage = useCallback(async () => {
    const file = await getFile(IMAGE_KEY);
    if (file) {
      setPreviewFromFile(file);
    }
  }, [setPreviewFromFile]);

  return {
    selectedFile,
    previewUrl,
    handleImageUpload,
    removeImage,
    restoreImage,
    setPreviewFromFile,
  };
}
