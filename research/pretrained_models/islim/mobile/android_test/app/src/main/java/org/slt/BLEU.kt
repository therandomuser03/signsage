package org.slt

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.min

data class BleuResult(
    val psB1: Double,
    val psB2: Double,
    val psB3: Double,
    val psB4: Double,
)

object BLEU {
    private fun getNgrams(tokens: List<String>, maxOrder: Int): Map<List<String>, Int> {
        val counts = mutableMapOf<List<String>, Int>()
        for (order in 1..maxOrder) {
            for (i in 0..tokens.size - order) {
                val ng = tokens.subList(i, i + order)
                counts[ng] = counts.getOrDefault(ng, 0) + 1
            }
        }
        return counts
    }

    fun compute(predictions: List<String>, references: List<String>): BleuResult {
        // Tokenize by whitespace
        val predTok = predictions.map { it.trim().split("\\s+".toRegex()).filter { s -> s.isNotEmpty() } }
        val refTok = references.map { listOf(it.trim().split("\\s+".toRegex()).filter { s -> s.isNotEmpty() }) }

        val maxN = 4
        val matches = IntArray(4)
        val possible = IntArray(4)
        var refLen = 0
        var transLen = 0

        for (i in predTok.indices) {
            val refLenI = refTok[i].minOfOrNull { it.size } ?: 0
            refLen += refLenI
            transLen += predTok[i].size

            // Merge n-grams across all references for this item
            val merged = mutableMapOf<List<String>, Int>()
            for (r in refTok[i]) {
                val ngrams = getNgrams(r, maxN)
                for ((ng, cnt) in ngrams) {
                    merged[ng] = maxOf(merged.getOrDefault(ng, 0), cnt)
                }
            }

            // Count clipped matches for pred
            val predNgrams = getNgrams(predTok[i], maxN)
            for ((ng, cnt) in predNgrams) {
                val clipped = minOf(cnt, merged.getOrDefault(ng, 0))
                matches[ng.size - 1] += clipped
            }

            // Count possible
            for (n in 1..maxN) {
                possible[n - 1] += maxOf(0, predTok[i].size - n + 1)
            }
        }

        // Smooth + compute precisions
        val precs = DoubleArray(maxN)
        for (n in 0 until maxN) {
            precs[n] = (matches[n] + 1.0) / (possible[n] + 1.0)
        }

        val geoMean = if (precs.any { it <= 0.0 }) 0.0
            else exp(precs.sumOf { ln(it) } / maxN)

        val bp = if (transLen >= refLen) 1.0
            else if (transLen > 0) exp(1.0 - refLen.toDouble() / transLen)
            else 0.0

        // Cumulative BLEU = geometric mean up to each order × brevity penalty
        val b1 = String.format("%.2f", precs[0] * bp * 100).toDouble()
        val b2 = String.format("%.2f", exp((ln(precs[0]) + ln(precs[1])) / 2) * bp * 100).toDouble()
        val b3 = String.format("%.2f", exp((ln(precs[0]) + ln(precs[1]) + ln(precs[2])) / 3) * bp * 100).toDouble()
        val b4 = String.format("%.2f", geoMean * bp * 100).toDouble()

        return BleuResult(b1, b2, b3, b4)
    }
}
