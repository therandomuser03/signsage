package org.slt

import android.os.Bundle
import android.util.Log
import android.app.Activity
import android.widget.ScrollView
import android.widget.TextView
import android.graphics.Color
import android.graphics.Typeface
import android.util.TypedValue
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File

/**
 * Runs the 57 iSign test samples through the exported TorchScript model
 * and prints PS-BLEU scores to logcat and on-screen console.
 *
 * Usage:
 *   adb logcat -s SLTTest
 */
class TestActivity : Activity() {

    private val tag = "SLTTest"
    private lateinit var modelDir: File
    private lateinit var testDir: File
    private lateinit var console: TextView

    // Models
    private lateinit var encoder: Module
    private lateinit var t5Encoder: Module
    private lateinit var firstStep: Module
    private lateinit var decoderStep: Module
    private lateinit var lmHead: Module

    private var vocabSize = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Simple UI to show logs
        val scrollView = ScrollView(this)
        console = TextView(this).apply {
            setBackgroundColor(Color.BLACK)
            setTextColor(Color.GREEN)
            typeface = Typeface.MONOSPACE
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            setPadding(16, 16, 16, 16)
        }
        scrollView.addView(console)
        setContentView(scrollView)

        log("App started")

        // Copy assets to internal storage (Module.load() needs file paths)
        modelDir = File(filesDir, "slt_models").also { it.mkdirs() }
        testDir = File(filesDir, "test_data").also { it.mkdirs() }
        
        Thread {
            log("Copying assets...")
            copyAssetDir("slt_models", modelDir)
            copyAssetDir("test_data", testDir)
            log("Assets copied")
            runTest()
        }.start()
    }

    private fun log(msg: String) {
        Log.i(tag, msg)
        if (msg.contains("==") || msg.contains("✅") || msg.contains("[") || msg.contains("Results") || msg.contains("──")) {
            runOnUiThread {
                console.append("$msg\n")
            }
        }
    }

    private fun runTest() {
        log("========================================")
        log("SLT Android Test — 57 iSign Intersection")
        log("========================================")

        // ── 1. Load models ──
        try {
            log("Loading models (Lite)...")
            encoder = LiteModuleLoader.load(File(modelDir, "encoder_traced.ptl").absolutePath)
            t5Encoder = LiteModuleLoader.load(File(modelDir, "t5_encoder.ptl").absolutePath)
            firstStep = LiteModuleLoader.load(File(modelDir, "t5_decoder_first_step.ptl").absolutePath)
            decoderStep = LiteModuleLoader.load(File(modelDir, "t5_decoder_step.ptl").absolutePath)
            lmHead = LiteModuleLoader.load(File(modelDir, "t5_lm_head.ptl").absolutePath)
            log("✅ Models loaded")
        } catch (e: Exception) {
            log("❌ Failed to load models: ${e.message}")
            return
        }

        // ── 2. Load test data ──
        val references: List<String>
        val vocab: Map<Int, String>
        val kpModule: Module
        try {
            log("Loading test data...")
            kpModule = LiteModuleLoader.load(File(testDir, "keypoints_module.ptl").absolutePath)
            references = File(testDir, "references.txt").readLines()
            vocab = loadVocab()
            log("✅ Test data loaded: ${references.size} samples, vocab: ${vocab.size}")
            if (vocab.isEmpty()) log("⚠️ Warning: Vocab is empty!")
        } catch (e: Exception) {
            log("❌ Failed to load test data: ${e.message}")
            return
        }

        // ── 3. Run inference ──
        val numSamples = references.size
        val predictions = mutableListOf<String>()
        val timings = mutableListOf<Long>()

        log("── Running inference on $numSamples samples ──")

        for (i in 0 until numSamples) {
            val t0 = System.currentTimeMillis()

            try {
                // Extract single sample keypoints (1, 4, 94, 128)
                val sampleKp = kpModule.forward(IValue.from(i.toLong())).toTensor()
                val prediction = translate(sampleKp, vocab)

                val elapsed = System.currentTimeMillis() - t0
                timings.add(elapsed)

                predictions.add(prediction)
                log("[${i + 1}/$numSamples] ${elapsed}ms")
                log("  Ref: ${references[i]}")
                log("  Pred: $prediction")
                log("  Match: ${if (prediction == references[i]) "✅" else "❌"}")
            } catch (e: Exception) {
                log("❌ Error at sample ${i+1}: ${e.message}")
            }

            System.gc()
            Thread.sleep(100)
        }

        // ── 4. Compute BLEU ──
        val avgTime = timings.average()
        val tokenMatch = predictions.zip(references).count { (p, r) -> p == r }
        val bleu = BLEU.compute(predictions, references)

        log("═══════════════════════════════════════")
        log("Results ($numSamples samples):")
        log("  Average time: ${"%.0f".format(avgTime)}ms per sample")
        log("  Total time: ${timings.sum()}ms")
        log("  Token-match: $tokenMatch/$numSamples = " +
                "${"%.1f".format(tokenMatch.toDouble() / numSamples * 100)}%")
        log("  PS-BLEU-1: ${bleu.psB1}")
        log("  PS-BLEU-2: ${bleu.psB2}")
        log("  PS-BLEU-3: ${bleu.psB3}")
        log("  PS-BLEU-4: ${bleu.psB4}")
        log("═══════════════════════════════════════")

        // Also show on screen for convenience
        runOnUiThread {
            android.widget.Toast.makeText(
                this,
                "PS-BLEU-4: ${bleu.psB4}  (${"%.0f".format(avgTime)}ms/sample)",
                android.widget.Toast.LENGTH_LONG
            ).show()
        }
    }

    private fun translate(kp: Tensor, vocab: Map<Int, String>): String {
        // Encoder
        val visOut = encoder.forward(IValue.from(kp)).toTensor()
        val t5Out = t5Encoder.forward(IValue.from(visOut)).toTensor()

        // Greedy decode
        val padId = 0L
        val eosId = 1L
        val maxLen = 64

        val tokenIds = mutableListOf(padId)

        // First step
        val padTensor = Tensor.fromBlob(longArrayOf(padId), longArrayOf(1, 1))
        val firstOut = firstStep.forward(IValue.from(t5Out), IValue.from(padTensor)).toTuple()
        var hidden = firstOut[0].toTensor()
        var saK = firstOut[1]
        var saV = firstOut[2]
        var caK = firstOut[3]
        var caV = firstOut[4]

        // Apply lm_head
        var logits = lmHead.forward(IValue.from(hidden)).toTensor()
        var nextId = argmax(logits)
        tokenIds.add(nextId)

        if (nextId == eosId) return decodeIds(tokenIds, vocab)

        // Subsequent steps
        for (step in 1 until maxLen) {
            val idsTensor = Tensor.fromBlob(longArrayOf(nextId), longArrayOf(1, 1))
            val out = decoderStep.forward(
                IValue.from(t5Out),
                IValue.from(idsTensor),
                saK, saV, caK, caV
            ).toTuple()

            logits = out[0].toTensor()
            saK = out[1]
            saV = out[2]
            caK = out[3]
            caV = out[4]

            nextId = argmax(logits)
            tokenIds.add(nextId)
            if (nextId == eosId) break
        }

        return decodeIds(tokenIds, vocab)
    }

    private fun argmax(t: Tensor): Long {
        val data = t.dataAsFloatArray
        var maxIdx = 0
        var maxVal = data[0]
        for (i in 1 until data.size) {
            if (data[i] > maxVal) {
                maxVal = data[i]
                maxIdx = i
            }
        }
        return maxIdx.toLong()
    }

    private fun decodeIds(ids: List<Long>, vocab: Map<Int, String>): String {
        val tokens = mutableListOf<String>()
        for (id in ids) {
            if (id == 0L || id == 1L) continue // skip pad/eos
            val t = vocab[id.toInt()]
            if (t == null) {
                tokens.add("<id:$id>")
            } else {
                tokens.add(t)
            }
        }
        return tokens.joinToString("")
            .replace("\u2581", " ")
            .replace("\\u2581", " ")
            .trim()
            .replace("\\s+".toRegex(), " ")
    }

    private fun loadVocab(): Map<Int, String> {
        val map = mutableMapOf<Int, String>()
        val json = File(testDir, "vocab.json").readText()
            .trim('{', '}', ' ', '\n')
        if (json.isEmpty()) return map
        for (entry in json.split(",")) {
            val parts = entry.trim().split(":", limit = 2)
            if (parts.size == 2) {
                val id = parts[0].trim().trim('"').toIntOrNull() ?: continue
                val token = parts[1].trim().trim('"')
                if (token.isNotEmpty()) map[id] = token
            }
        }
        return map
    }

    private fun copyAssetDir(assetSubDir: String, destDir: File) {
        val files = assets.list(assetSubDir) ?: return
        for (fname in files) {
            val assetPath = "$assetSubDir/$fname"
            val destFile = File(destDir, fname)
            if (destFile.exists() && destFile.length() > 0) {
                // log("Skipping $fname (exists)")
                continue
            }
            try {
                assets.open(assetPath).use { input ->
                    destFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            } catch (e: Exception) {
                Log.w(tag, "Failed to copy $assetPath: ${e.message}")
            }
        }
    }
}
