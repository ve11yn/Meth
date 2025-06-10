// import { useState, useRef } from "react";
// import { Upload, FileImage, CheckCircle, AlertCircle, Loader2, Calculator, RotateCcw } from "lucide-react";

// const API_BASE_URL = 'http://localhost:5001';

// type UploadResult = {
//   success?: boolean;
//   equation?: string;
//   parsed_equation?: string;
//   result?: string | number | null;
//   full_calculation?: string;
//   image_info?: {
//     format?: string;
//     width?: number;
//     height?: number;
//   };
//   details?: {
//     character_count?: number;
//   };
//   filename?: string;
//   error?: string;
//   message?: string;
// };

// const FileUploader = () => {
//   const [file, setFile] = useState<File | null>(null);
//   const [uploading, setUploading] = useState(false);
//   const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
//   const [error, setError] = useState<string | null>(null);
//   const [imagePreview, setImagePreview] = useState<string | null>(null);
//   const fileInputRef = useRef<HTMLInputElement | null>(null);

//   const BACKEND_URL = import.meta.env.VITE_API_URL || API_BASE_URL;

//   const handleFileChange = (e) => {
//     if (e.target.files && e.target.files.length > 0) {
//       const selectedFile = e.target.files[0];
//       setFile(selectedFile);
//       setError(null);
//       setUploadResult(null);
      
//       // Create image preview
//       const reader = new FileReader();
//       reader.onload = (e) => {
//         setImagePreview(e.target?.result as string);
//       };
//       reader.readAsDataURL(selectedFile);
//     }
//   };

//   const handleDrop = (e) => {
//     e.preventDefault();
//     if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
//       const droppedFile = e.dataTransfer.files[0];
//       setFile(droppedFile);
//       setError(null);
//       setUploadResult(null);
      
//       // Create image preview
//       const reader = new FileReader();
//       reader.onload = (e) => {
//         setImagePreview(e.target?.result as string);
//       };
//       reader.readAsDataURL(droppedFile);
//     }
//   };

//   const resetForNewProblem = () => {
//     setFile(null);
//     setError(null);
//     setUploadResult(null);
//     setImagePreview(null);
//     if (fileInputRef.current) {
//       fileInputRef.current.value = '';
//     }
//   };

//   const handleUpload = async () => {
//     if (!file) return;

//     // Validate file type and size before uploading
//     const allowedTypes = ['image/png', 'image/jpg', 'image/jpeg', 'image/webp', 'image/heic'];
//     if (!allowedTypes.includes(file.type)) {
//       setError('File type not supported. Please use PNG, JPG, JPEG, WebP, or HEIC.');
//       return;
//     }

//     const maxSize = 18 * 1024 * 1024; // 18MB
//     if (file.size > maxSize) {
//       setError('File size too large. Maximum size is 18MB.');
//       return;
//     }

//     setUploading(true);
//     setError(null);

//     try {
//       const formData = new FormData();
//       formData.append("file", file);

//       console.log(`Uploading to: ${BACKEND_URL}/upload`);

//       const response = await fetch(`${BACKEND_URL}/upload`, {
//         method: "POST",
//         body: formData,
//       });

//       const data = await response.json();

//       if (response.ok && data.success) {
//         setUploadResult(data);
//         console.log("Upload successful:", data);
//       } else {
//         setError(data.error || data.message || "Upload failed");
//       }
//     } catch (err) {
//       console.error("Upload error:", err);
//       if (err instanceof TypeError && err.message.includes('fetch')) {
//         setError("Unable to connect to server. Make sure the backend is running on port 5001.");
//       } else {
//         setError("Network error: Unable to connect to server");
//       }
//     } finally {
//       setUploading(false);
//     }
//   };

//   const formatResult = (result) => {
//     if (typeof result === 'number') {
//       return result.toString();
//     }
//     return result;
//   };

//   return (
//     <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 max-w-5xl mx-auto mt-10">
//       <div className="bg-gray-900/50 border border-gray-800 backdrop-blur-sm rounded-lg flex flex-col">
//         <div className="p-4 flex-1 flex flex-col">
//           <div className="flex items-center justify-between mb-3">
//             <h2 className="text-lg font-semibold text-white flex items-center">
//               <Upload className="mr-2 h-4 w-4" />
//               Upload Math Problem
//             </h2>
//             {(file || uploadResult) && (
//               <button
//                 onClick={resetForNewProblem}
//                 className="text-gray-400 hover:text-white transition-colors p-1.5 rounded-md hover:bg-gray-800"
//                 title="Start over"
//               >
//                 <RotateCcw className="h-4 w-4" />
//               </button>
//             )}
//           </div>

//           {!imagePreview ? (
//             <div
//               className="flex-1 border-2 border-dashed border-gray-700 rounded-xl p-4 text-center hover:border-gray-600 transition-colors cursor-pointer flex flex-col items-center justify-center min-h-[200px] max-h-[300px]"
//               onClick={() => fileInputRef.current?.click()}
//               onDrop={handleDrop}
//               onDragOver={(e) => e.preventDefault()}
//             >
//               <div className="w-12 h-12 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-3">
//                 <Upload className="h-6 w-6 text-gray-400" />
//               </div>
//               <h3 className="text-lg font-light mb-2 text-white">
//                 Drop your math problem here
//               </h3>
//               <p className="text-gray-400 mb-4 text-sm">
//                 or click to browse files
//               </p>
//               <button
//                 type="button"
//                 className="border border-gray-700 text-gray-300 hover:bg-gray-800 bg-transparent px-3 py-2 rounded-md inline-flex items-center transition-colors text-sm"
//                 disabled={uploading}
//               >
//                 <FileImage className="mr-2 h-4 w-4" />
//                 Choose Image
//               </button>
//             </div>
//           ) : (
//             <div className="flex-1 flex flex-col">
//               <div className="flex-1 bg-gray-800/30 rounded-xl p-3 flex items-center justify-center overflow-hidden min-h-[200px]">
//                 <img
//                   src={imagePreview}
//                   alt="Math problem preview"
//                   className="max-w-full max-h-full object-contain rounded-lg"
//                 />
//               </div>
//               {file && (
//                 <div className="mt-2 text-xs text-gray-300">
//                   <p className="font-medium truncate">{file.name}</p>
//                   <p className="text-xs text-gray-500">
//                     Size: {(file.size / 1024 / 1024).toFixed(2)} MB
//                   </p>
//                 </div>
//               )}
//             </div>
//           )}

//           <input
//             ref={fileInputRef}
//             type="file"
//             accept="image/png,image/jpg,image/jpeg,image/webp,image/heic"
//             onChange={handleFileChange}
//             className="hidden"
//           />

//           {/* File Format Info */}
//           <div className="mt-2 text-center">
//             <p className="text-xs text-gray-500">
//               Supports: JPG, PNG, HEIC, WebP (Max 18MB)
//             </p>
//           </div>

//           {file && !uploadResult && (
//             <div className="mt-3">
//               <button
//                 onClick={handleUpload}
//                 disabled={uploading}
//                 className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:cursor-not-allowed text-white px-4 py-2.5 rounded-md transition-colors font-medium inline-flex items-center justify-center text-sm"
//               >
//                 {uploading ? (
//                   <>
//                     <Loader2 className="mr-2 h-4 w-4 animate-spin" />
//                     Processing...
//                   </>
//                 ) : (
//                   <>
//                     <Calculator className="mr-2 h-4 w-4" />
//                     Solve Equation
//                   </>
//                 )}
//               </button>
//             </div>
//           )}

//           {error && (
//             <div className="mt-3 p-3 bg-red-900/50 border border-red-700 rounded-md">
//               <div className="flex items-start">
//                 <AlertCircle className="h-4 w-4 text-red-400 mr-2 mt-0.5 flex-shrink-0" />
//                 <p className="text-red-300 text-xs">{error}</p>
//               </div>
//             </div>
//           )}
//         </div>
//       </div>

//       <div className="bg-gray-900/50 border border-gray-800 backdrop-blur-sm rounded-lg flex flex-col">
//         <div className="p-4 flex-1 flex flex-col">
//           <h2 className="text-lg font-semibold text-white mb-3 flex items-center">
//             <Calculator className="mr-2 h-4 w-4" />
//             Solution
//           </h2>

//           {!uploadResult ? (
//             <div className="flex-1 flex flex-col items-center justify-center text-center min-h-[300px]">
//               <div className="w-12 h-12 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-3">
//                 <Calculator className="h-6 w-6 text-gray-400" />
//               </div>
//               <p className="text-gray-400 text-base">
//                 Get your equations solved instantly
//               </p>
//             </div>
//           ) : (
//             <div className="flex-1 space-y-4">
//               <div className="p-4 bg-gradient-to-r from-green-900/50 to-blue-900/50 border border-green-700 rounded-lg">
//                 <div className="flex items-center mb-3">
//                   <CheckCircle className="h-5 w-5 text-green-400 mr-2" />
//                   <h3 className="text-green-300 font-semibold">
//                     Equation Solved!
//                   </h3>
//                 </div>

//                 {uploadResult.equation && (
//                   <div className="mb-3">
//                     <p className="text-green-300 font-medium mb-2 text-sm">
//                       Detected Equation:
//                     </p>
//                     <div className="bg-gray-800/50 rounded-md p-2 border border-gray-600">
//                       <span className="font-mono text-white text-base">
//                         {uploadResult.parsed_equation || uploadResult.equation}
//                       </span>
//                     </div>
//                   </div>
//                 )}

//                 {uploadResult.result !== undefined && uploadResult.result !== null && (
//                   <div className="mb-3">
//                     <p className="text-green-300 font-medium mb-2 text-sm">Answer:</p>
//                     <div className="bg-gradient-to-r from-green-800/50 to-blue-800/50 rounded-lg p-3 border-2 border-green-500/50">
//                       <div className="text-center">
//                         <span className="font-mono text-2xl font-bold text-white">
//                           {formatResult(uploadResult.result)}
//                         </span>
//                       </div>
//                     </div>
//                   </div>
//                 )}

//                 {uploadResult.full_calculation &&
//                   uploadResult.full_calculation !== uploadResult.result && (
//                     <div>
//                       <p className="text-green-300 font-medium mb-2 text-sm">
//                         Calculation Steps:
//                       </p>
//                       <div className="bg-gray-800/30 rounded-md p-2 border border-gray-600">
//                         <span className="font-mono text-gray-200 text-xs">
//                           {uploadResult.full_calculation}
//                         </span>
//                       </div>
//                     </div>
//                   )}
//               </div>

//               <div className="p-3 bg-gray-800/30 border border-gray-700 rounded-md">
//                 <h4 className="text-gray-300 font-medium mb-2 text-sm">
//                   Processing Details
//                 </h4>

//                 <div className="grid grid-cols-2 gap-3 text-xs">
//                   {uploadResult.image_info && (
//                     <div className="space-y-1">
//                       <p className="text-gray-400">Image Info:</p>
//                       <p className="text-gray-300">Format: {uploadResult.image_info.format}</p>
//                       <p className="text-gray-300">
//                         Size: {uploadResult.image_info.width} × {uploadResult.image_info.height}
//                       </p>
//                     </div>
//                   )}

//                   {uploadResult.details && (
//                     <div className="space-y-1">
//                       <p className="text-gray-400">Analysis:</p>
//                       <p className="text-gray-300">
//                         Characters: {uploadResult.details.character_count || 0}
//                       </p>
//                       {uploadResult.filename && (
//                         <p className="text-gray-300 truncate" title={uploadResult.filename}>
//                           File: {uploadResult.filename}
//                         </p>
//                       )}
//                     </div>
//                   )}
//                 </div>
//               </div>

              
//             </div>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// };

// export default FileUploader;


import { useState, useRef } from "react";
import { Upload, FileImage, CheckCircle, AlertCircle, Loader2, Calculator, RotateCcw } from "lucide-react";

// Updated API configuration for Vercel deployment
const getApiBaseUrl = () => {
  // In production (Vercel), use relative URLs (same domain)
  if (process.env.NODE_ENV === 'production') {
    return '';
  }
  // In development, use environment variable or fallback to localhost
  return import.meta.env.VITE_API_URL || 'http://localhost:5001';
};

type UploadResult = {
  success?: boolean;
  equation?: string;
  parsed_equation?: string;
  result?: string | number | null;
  full_calculation?: string;
  image_info?: {
    format?: string;
    width?: number;
    height?: number;
  };
  details?: {
    character_count?: number;
  };
  filename?: string;
  error?: string;
  message?: string;
};

const FileUploader = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const BACKEND_URL = getApiBaseUrl();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setError(null);
      setUploadResult(null);
      
      // Create image preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      setFile(droppedFile);
      setError(null);
      setUploadResult(null);
      
      // Create image preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(droppedFile);
    }
  };

  const resetForNewProblem = () => {
    setFile(null);
    setError(null);
    setUploadResult(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    // Validate file type and size before uploading
    const allowedTypes = ['image/png', 'image/jpg', 'image/jpeg', 'image/webp', 'image/heic'];
    if (!allowedTypes.includes(file.type)) {
      setError('File type not supported. Please use PNG, JPG, JPEG, WebP, or HEIC.');
      return;
    }

    const maxSize = 16 * 1024 * 1024; // 16MB (reduced for Vercel limits)
    if (file.size > maxSize) {
      setError('File size too large. Maximum size is 16MB.');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      // Updated endpoint to use /api/upload for Vercel
      const uploadUrl = `${BACKEND_URL}/api/upload`;
      console.log(`Uploading to: ${uploadUrl}`);

      const response = await fetch(uploadUrl, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok && data.success) {
        setUploadResult(data);
        console.log("Upload successful:", data);
      } else {
        setError(data.error || data.message || "Upload failed");
      }
    } catch (err) {
      console.error("Upload error:", err);
      if (err instanceof TypeError && err.message.includes('fetch')) {
        if (process.env.NODE_ENV === 'production') {
          setError("Unable to connect to the API. Please try again.");
        } else {
          setError("Unable to connect to server. Make sure the backend is running on port 5001.");
        }
      } else {
        setError("Network error: Unable to connect to server");
      }
    } finally {
      setUploading(false);
    }
  };

  // Health check function for debugging
  const checkApiHealth = async () => {
    try {
      const healthUrl = `${BACKEND_URL}/api/health`;
      const response = await fetch(healthUrl);
      const data = await response.json();
      console.log('API Health:', data);
    } catch (err) {
      console.error('Health check failed:', err);
    }
  };

  const formatResult = (result: string | number | null | undefined) => {
    if (typeof result === 'number') {
      return result.toString();
    }
    return result || '';
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 max-w-5xl mx-auto mt-10">
      <div className="bg-gray-900/50 border border-gray-800 backdrop-blur-sm rounded-lg flex flex-col">
        <div className="p-4 flex-1 flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold text-white flex items-center">
              <Upload className="mr-2 h-4 w-4" />
              Upload Math Problem
            </h2>
            <div className="flex items-center gap-2">
              {/* Debug button for development */}
              {process.env.NODE_ENV === 'development' && (
                <button
                  onClick={checkApiHealth}
                  className="text-gray-400 hover:text-white transition-colors p-1.5 rounded-md hover:bg-gray-800 text-xs"
                  title="Check API Health"
                >
                  🔍
                </button>
              )}
              {(file || uploadResult) && (
                <button
                  onClick={resetForNewProblem}
                  className="text-gray-400 hover:text-white transition-colors p-1.5 rounded-md hover:bg-gray-800"
                  title="Start over"
                >
                  <RotateCcw className="h-4 w-4" />
                </button>
              )}
            </div>
          </div>

          {!imagePreview ? (
            <div
              className="flex-1 border-2 border-dashed border-gray-700 rounded-xl p-4 text-center hover:border-gray-600 transition-colors cursor-pointer flex flex-col items-center justify-center min-h-[200px] max-h-[300px]"
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
            >
              <div className="w-12 h-12 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-3">
                <Upload className="h-6 w-6 text-gray-400" />
              </div>
              <h3 className="text-lg font-light mb-2 text-white">
                Drop your math problem here
              </h3>
              <p className="text-gray-400 mb-4 text-sm">
                or click to browse files
              </p>
              <button
                type="button"
                className="border border-gray-700 text-gray-300 hover:bg-gray-800 bg-transparent px-3 py-2 rounded-md inline-flex items-center transition-colors text-sm"
                disabled={uploading}
              >
                <FileImage className="mr-2 h-4 w-4" />
                Choose Image
              </button>
            </div>
          ) : (
            <div className="flex-1 flex flex-col">
              <div className="flex-1 bg-gray-800/30 rounded-xl p-3 flex items-center justify-center overflow-hidden min-h-[200px]">
                <img
                  src={imagePreview}
                  alt="Math problem preview"
                  className="max-w-full max-h-full object-contain rounded-lg"
                />
              </div>
              {file && (
                <div className="mt-2 text-xs text-gray-300">
                  <p className="font-medium truncate">{file.name}</p>
                  <p className="text-xs text-gray-500">
                    Size: {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              )}
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpg,image/jpeg,image/webp,image/heic"
            onChange={handleFileChange}
            className="hidden"
          />

          {/* File Format Info */}
          <div className="mt-2 text-center">
            <p className="text-xs text-gray-500">
              Supports: JPG, PNG, HEIC, WebP (Max 16MB)
            </p>
          </div>

          {file && !uploadResult && (
            <div className="mt-3">
              <button
                onClick={handleUpload}
                disabled={uploading}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:cursor-not-allowed text-white px-4 py-2.5 rounded-md transition-colors font-medium inline-flex items-center justify-center text-sm"
              >
                {uploading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Calculator className="mr-2 h-4 w-4" />
                    Solve Equation
                  </>
                )}
              </button>
            </div>
          )}

          {error && (
            <div className="mt-3 p-3 bg-red-900/50 border border-red-700 rounded-md">
              <div className="flex items-start">
                <AlertCircle className="h-4 w-4 text-red-400 mr-2 mt-0.5 flex-shrink-0" />
                <p className="text-red-300 text-xs">{error}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="bg-gray-900/50 border border-gray-800 backdrop-blur-sm rounded-lg flex flex-col">
        <div className="p-4 flex-1 flex flex-col">
          <h2 className="text-lg font-semibold text-white mb-3 flex items-center">
            <Calculator className="mr-2 h-4 w-4" />
            Solution
          </h2>

          {!uploadResult ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center min-h-[300px]">
              <div className="w-12 h-12 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-3">
                <Calculator className="h-6 w-6 text-gray-400" />
              </div>
              <p className="text-gray-400 text-base">
                Get your equations solved instantly
              </p>
              <p className="text-gray-500 text-sm mt-2">
                Upload a handwritten math problem to see the solution
              </p>
            </div>
          ) : (
            <div className="flex-1 space-y-4">
              <div className="p-4 bg-gradient-to-r from-green-900/50 to-blue-900/50 border border-green-700 rounded-lg">
                <div className="flex items-center mb-3">
                  <CheckCircle className="h-5 w-5 text-green-400 mr-2" />
                  <h3 className="text-green-300 font-semibold">
                    Equation Solved!
                  </h3>
                </div>

                {uploadResult.equation && (
                  <div className="mb-3">
                    <p className="text-green-300 font-medium mb-2 text-sm">
                      Detected Equation:
                    </p>
                    <div className="bg-gray-800/50 rounded-md p-2 border border-gray-600">
                      <span className="font-mono text-white text-base">
                        {uploadResult.parsed_equation || uploadResult.equation}
                      </span>
                    </div>
                  </div>
                )}

                {uploadResult.result !== undefined && uploadResult.result !== null && (
                  <div className="mb-3">
                    <p className="text-green-300 font-medium mb-2 text-sm">Answer:</p>
                    <div className="bg-gradient-to-r from-green-800/50 to-blue-800/50 rounded-lg p-3 border-2 border-green-500/50">
                      <div className="text-center">
                        <span className="font-mono text-2xl font-bold text-white">
                          {formatResult(uploadResult.result)}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {uploadResult.full_calculation &&
                  uploadResult.full_calculation !== uploadResult.result && (
                    <div>
                      <p className="text-green-300 font-medium mb-2 text-sm">
                        Calculation Steps:
                      </p>
                      <div className="bg-gray-800/30 rounded-md p-2 border border-gray-600">
                        <span className="font-mono text-gray-200 text-xs">
                          {uploadResult.full_calculation}
                        </span>
                      </div>
                    </div>
                  )}
              </div>

              <div className="p-3 bg-gray-800/30 border border-gray-700 rounded-md">
                <h4 className="text-gray-300 font-medium mb-2 text-sm">
                  Processing Details
                </h4>

                <div className="grid grid-cols-2 gap-3 text-xs">
                  {uploadResult.image_info && (
                    <div className="space-y-1">
                      <p className="text-gray-400">Image Info:</p>
                      <p className="text-gray-300">Format: {uploadResult.image_info.format}</p>
                      <p className="text-gray-300">
                        Size: {uploadResult.image_info.width} × {uploadResult.image_info.height}
                      </p>
                    </div>
                  )}

                  {uploadResult.details && (
                    <div className="space-y-1">
                      <p className="text-gray-400">Analysis:</p>
                      <p className="text-gray-300">
                        Characters: {uploadResult.details.character_count || 0}
                      </p>
                      {uploadResult.filename && (
                        <p className="text-gray-300 truncate" title={uploadResult.filename}>
                          File: {uploadResult.filename}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FileUploader;