package org.slt

import android.content.Context
import android.util.Log
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader

/**
 * JNI wrapper for the SLT C++ native library.
 *
 * Usage:
 *   SLTModule.init(context)  // copies models from assets to internal storage
 *   val text = SLTModule.translate(keypoints, numFrames)
 *   SLTModule.destroy()
 */
object SLTModule {

    private const val TAG = "SLTModule"
    private var initialized = false
    private var vocab: Map<String, Int> = emptyMap()
    private var idToToken: Map<Int, String> = emptyMap()
    private var padId = 0
    private var eosId = 1

    // ── JNI ──
    private external fun initModel(modelDir: String): Boolean
    private external fun translateNative(keypoints: FloatArray, numFrames: Int): String
    private external fun destroyModel()

    // ── Public API ──

    fun init(context: Context) {
        if (initialized) return

        // Copy models from assets to internal storage (libtorch loads from file system)
        val modelDir = File(context.filesDir, "slt_models")
        copyAssets(context, modelDir)

        System.loadLibrary("slt_jni")
        val ok = initModel(modelDir.absolutePath)
        if (!ok) throw RuntimeException("SLTModel init failed")

        // Load tokenizer vocab
        loadTokenizer(context, modelDir)

        initialized = true
    }

    /**
     * Translate keypoints to text.
     * @param keypoints flat float array in (4, 94, numFrames) layout
     * @param numFrames number of frames
     * @return translated text
     */
    fun translate(keypoints: FloatArray, numFrames: Int): String {
        if (!initialized) throw IllegalStateException("SLTModule not initialized")
        val idsCsv = translateNative(keypoints, numFrames)
        return decodeIds(idsCsv)
    }

    fun destroy() {
        if (initialized) {
            destroyModel()
            initialized = false
        }
    }

    // ── Private ──

    private fun decodeIds(idsCsv: String): String {
        if (idsCsv.isEmpty()) return ""
        val ids = idsCsv.split(",").map { it.toIntOrNull() ?: padId }
        val tokens = mutableListOf<String>()
        for (id in ids) {
            if (id == padId || id == eosId) continue
            tokens.add(idToToken[id] ?: "<unk>")
        }
        return tokens.joinToString(" ")
    }

    private fun loadTokenizer(context: Context, modelDir: File) {
        // Try loading from model dir first (copied from assets)
        val vocabFile = File(modelDir, "vocab.json")
        if (vocabFile.exists()) {
            val json = vocabFile.readText()
            // Parse simple JSON: {"token": id, ...}
            val entries = json.trim('{', '}', ' ', '\n').split(",")
            for (entry in entries) {
                val parts = entry.trim().split(":")
                if (parts.size == 2) {
                    val token = parts[0].trim().trim('"')
                    val id = parts[1].trim().toIntOrNull() ?: continue
                    vocab = vocab + (token to id)
                    idToToken = idToToken + (id to token)
                }
            }
        }
        // T5 uses SentencePiece. For simplicity, use a basic fallback.
        // In production, use a SentencePiece Java binding or the T5 tokenizer in Java.
        if (vocab.isEmpty()) {
            Log.w(TAG, "vocab.json not found, using identity fallback")
            // Fallback: assume token IDs map roughly (won't produce correct text)
            // This is a placeholder — real app needs FullT5Tokenizer or SentencePiece
        }
        padId = idToToken.entries.firstOrNull { it.value == "<pad>" }?.key ?: 0
        eosId = idToToken.entries.firstOrNull { it.value == "</s>" }?.key ?: 1
    }

    private fun copyAssets(context: Context, destDir: File) {
        if (destDir.exists()) return
        destDir.mkdirs()
        val assetFiles = context.assets.list("slt_models") ?: return
        for (fname in assetFiles) {
            context.assets.open("slt_models/$fname").use { input ->
                File(destDir, fname).outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
    }
}
