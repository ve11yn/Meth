import { Sparkles } from "lucide-react"

const Demo = () => {
    return (
        <div className="max-w-7xl mt-30 mx-auto"> {/* Wider max width */}
            <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-8 w-4xl border border-white/20">
                <div className="bg-white rounded-xl p-3 text-left">
                    <div className="flex items-center gap-2 mb-4">
                        <div className="w-1.5 h-1.5 bg-red-500 rounded-full"></div>
                        <div className="w-1.5 h-1.5 bg-yellow-500 rounded-full"></div>
                        <div className="w-1.5 h-1.5  bg-green-500 rounded-full"></div>
                        <span className="ml-4 text-sm text-gray-500">mathsolver.ai</span>
                    </div>
                    <div className="space-y-3 text-gray-800">
                        <div className="font-mono text-sm">
                            <span className="text-blue-600">Problem:</span> 2x + 5 = 15
                        </div>
                        <div className="font-mono text-sm">
                            <span className="text-green-600">Step 1:</span> 2x = 15 - 5
                        </div>
                        <div className="font-mono text-sm">
                            <span className="text-green-600">Step 2:</span> 2x = 10
                        </div>
                        <div className="font-mono text-sm">
                            <span className="text-purple-600">Answer:</span> x = 5
                        </div>
                        <div className="text-xs text-gray-500 mt-4 flex items-center gap-2">
                            <Sparkles className="h-3 w-3" />
                            AI solved this problem in 0.3 seconds
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Demo;


