package main

import (
	"encoding/binary"
	"math"
)

type GGUFType int

const (
	GGUFTypeF32  GGUFType = 0
	GGUFTypeF16  GGUFType = 1
	GGUFTypeQ4_0 GGUFType = 2
	GGUFTypeQ4_1 GGUFType = 3
	GGUFTypeQ5_0 GGUFType = 6
	GGUFTypeQ5_1 GGUFType = 7
	GGUFTypeQ8_0 GGUFType = 8
	GGUFTypeQ8_1 GGUFType = 9
	GGUFTypeQ2_K GGUFType = 10
	GGUFTypeQ3_K GGUFType = 11
	GGUFTypeQ4_K GGUFType = 12
	GGUFTypeQ5_K GGUFType = 13
	GGUFTypeQ6_K GGUFType = 14
)

func fp16ToFp32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := (h >> 10) & 0x1F
	mant := h & 0x3FF
	var f uint32

	if exp == 0 {
		if mant == 0 {
			f = sign
		} else {
			exp = 1
			for mant&0x400 == 0 {
				mant <<= 1
				exp--
			}
			mant &= 0x3FF
			f = sign | uint32(exp+127-15)<<23 | uint32(mant)<<13
		}
	} else if exp == 31 {
		f = sign | 0x7F800000 | uint32(mant)<<13
	} else {
		f = sign | uint32(exp+127-15)<<23 | uint32(mant)<<13
	}

	return math.Float32frombits(f)
}

func fp32ToFp16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := (bits >> 16) & 0x8000
	exp := int((bits>>23)&0xFF) - 127 + 15
	mant := bits & 0x7FFFFF

	if (bits>>23)&0xFF == 0 {
		return uint16(sign)
	}
	if (bits>>23)&0xFF == 0xFF {
		var m uint32
		if mant != 0 {
			m = 0x0200
		}
		return uint16(sign | 0x7C00 | m)
	}
	if exp >= 31 {
		return uint16(sign | 0x7C00)
	}
	if exp <= 0 {
		if exp < -10 {
			return uint16(sign)
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		roundBit := uint32(1) << (shift - 1)
		mant = (mant + roundBit) >> shift
		return uint16(sign | mant)
	}

	mant += 0x00001000
	if mant&0x00800000 != 0 {
		mant = 0
		exp++
		if exp >= 31 {
			return uint16(sign | 0x7C00)
		}
	}
	return uint16(sign | uint32(exp)<<10 | mant>>13)
}

func getScaleMinK4(j int, q []byte, sc, mn *byte) {
	if j < 4 {
		*sc = q[j] & 63
		*mn = q[j+4] & 63
	} else {
		*sc = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
		*mn = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
	}
}

// Dequantization functions

func dequantizeRowQ4K(src []byte, dst []float32, n int) {
	nb := n / 256
	offset := 0

	for i := 0; i < nb; i++ {
		d := fp16ToFp32(binary.LittleEndian.Uint16(src[offset:]))
		dmin := fp16ToFp32(binary.LittleEndian.Uint16(src[offset+2:]))
		scales := src[offset+4 : offset+16]
		qs := src[offset+16 : offset+144]
		y := dst[i*256:]

		is := 0
		qOff := 0
		for j := 0; j < 4; j++ {
			var sc, mn byte
			getScaleMinK4(is, scales, &sc, &mn)
			d1 := d * float32(sc)
			m1 := dmin * float32(mn)
			getScaleMinK4(is+1, scales, &sc, &mn)
			d2 := d * float32(sc)
			m2 := dmin * float32(mn)

			for l := 0; l < 32; l++ {
				y[l] = d1*float32(qs[qOff+l]&0xF) - m1
			}
			for l := 0; l < 32; l++ {
				y[l+32] = d2*float32(qs[qOff+l]>>4) - m2
			}
			y = y[64:]
			qOff += 32
			is += 2
		}
		offset += 144
	}
}

func dequantizeRowQ3K(src []byte, dst []float32, n int) {
	nb := n / 256
	offset := 0

	for i := 0; i < nb; i++ {
		d := fp16ToFp32(binary.LittleEndian.Uint16(src[offset:]))
		qs := src[offset+2 : offset+66]
		hmask := src[offset+66 : offset+98]
		scales := src[offset+98 : offset+110]

		var sc [16]int32
		for j := 0; j < 8; j++ {
			sc[j] = int32(scales[j] & 0xF)
		}
		for j := 0; j < 8; j++ {
			sc[8+j] = int32(scales[j] >> 4)
		}
		for j := 0; j < 4; j++ {
			sc[2*j] |= int32((scales[8+j]&3)<<4) | int32((scales[8+j]>>2)&3)<<4
			sc[2*j+1] |= int32((scales[8+j]>>2)&3) << 4
			sc[2*j+8] |= int32((scales[8+j]>>4)&3) << 4
			sc[2*j+9] |= int32((scales[8+j]>>6)&3) << 4
		}
		for j := 0; j < 16; j++ {
			sc[j] -= 32
		}

		outIdx := i * 256
		for j := 0; j < 256; j++ {
			q2 := (qs[j/4] >> (2 * (j % 4))) & 3
			hbit := (hmask[j/8] >> (j % 8)) & 1
			q3 := int(q2) | int(hbit<<2)
			sb := j / 16
			dst[outIdx+j] = d * float32(sc[sb]) * (float32(q3) - 4.0)
		}
		offset += 110
	}
}

func dequantizeRowQ2K(src []byte, dst []float32, n int) {
	nb := n / 256
	offset := 0

	for i := 0; i < nb; i++ {
		d := fp16ToFp32(binary.LittleEndian.Uint16(src[offset+16:]))
		dmin := fp16ToFp32(binary.LittleEndian.Uint16(src[offset+18:]))
		scales := src[offset : offset+16]
		qs := src[offset+16+4 : offset+16+4+64]

		outIdx := i * 256
		qOff := 0
		for j := 0; j < 256; j++ {
			q2 := (qs[qOff] >> (2 * (j % 4))) & 3
			if (j+1)%4 == 0 {
				qOff++
			}
			sb := j / 16
			sc := scales[sb] & 0xF
			mn := scales[sb] >> 4
			dst[outIdx+j] = d*float32(sc)*float32(q2) - dmin*float32(mn)
		}
		offset += 84
	}
}

func dequantizeRowQ6K(src []byte, dst []float32, n int) {
	nb := n / 256
	offset := 0

	for i := 0; i < nb; i++ {
		ql := src[offset : offset+128]
		qh := src[offset+128 : offset+192]
		sc := make([]int8, 16)
		for j := 0; j < 16; j++ {
			sc[j] = int8(src[offset+192+j])
		}
		d := fp16ToFp32(binary.LittleEndian.Uint16(src[offset+208:]))
		y := dst[i*256:]

		qlOff := 0
		qhOff := 0
		for chunk := 0; chunk < 256; chunk += 128 {
			is := chunk / 16
			for l := 0; l < 32; l++ {
				q1 := int((ql[qlOff+l]&0xF)|((qh[qhOff+l]>>0)&3)<<4) - 32
				q2 := int((ql[qlOff+l+32]&0xF)|((qh[qhOff+l]>>2)&3)<<4) - 32
				q3 := int((ql[qlOff+l]>>4)|((qh[qhOff+l]>>4)&3)<<4) - 32
				q4 := int((ql[qlOff+l+32]>>4)|((qh[qhOff+l]>>6)&3)<<4) - 32
				isL := is + (l / 16)
				y[l] = d * float32(sc[isL+0]) * float32(q1)
				y[l+32] = d * float32(sc[isL+2]) * float32(q2)
				y[l+64] = d * float32(sc[isL+4]) * float32(q3)
				y[l+96] = d * float32(sc[isL+6]) * float32(q4)
			}
			y = y[128:]
			qlOff += 64
			qhOff += 32
		}
		offset += 210
	}
}

func dequantizeRowQ8_0(src []byte, dst []float32, n int) {
	nb := n / 32
	offset := 0

	for i := 0; i < nb; i++ {
		d := fp16ToFp32(binary.LittleEndian.Uint16(src[offset:]))
		for j := 0; j < 32; j++ {
			dst[i*32+j] = d * float32(int8(src[offset+2+j]))
		}
		offset += 34
	}
}

func dequantizeRowQ4_0(src []byte, dst []float32, n int) {
	nb := n / 32
	offset := 0

	for i := 0; i < nb; i++ {
		d := fp16ToFp32(binary.LittleEndian.Uint16(src[offset:]))
		qs := src[offset+2 : offset+18]
		for j := 0; j < 32; j++ {
			var nibble byte
			if j < 16 {
				nibble = qs[j] & 0xF
			} else {
				nibble = qs[j-16] >> 4
			}
			dst[i*32+j] = d * (float32(nibble) - 8.0)
		}
		offset += 18
	}
}

func dequantizeRowF16(src []byte, dst []float32, n int) {
	for i := 0; i < n; i++ {
		dst[i] = fp16ToFp32(binary.LittleEndian.Uint16(src[i*2:]))
	}
}

func dequantizeRowF32(src []byte, dst []float32, n int) {
	for i := 0; i < n; i++ {
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
	}
}

func DequantizeRow(src []byte, dst []float32, n int, qtype GGUFType) {
	switch qtype {
	case GGUFTypeF32:
		dequantizeRowF32(src, dst, n)
	case GGUFTypeF16:
		dequantizeRowF16(src, dst, n)
	case GGUFTypeQ4_0:
		dequantizeRowQ4_0(src, dst, n)
	case GGUFTypeQ8_0:
		dequantizeRowQ8_0(src, dst, n)
	case GGUFTypeQ2_K:
		dequantizeRowQ2K(src, dst, n)
	case GGUFTypeQ3_K:
		dequantizeRowQ3K(src, dst, n)
	case GGUFTypeQ4_K:
		dequantizeRowQ4K(src, dst, n)
	case GGUFTypeQ6_K:
		dequantizeRowQ6K(src, dst, n)
	}
}

func GGUFTypeBlockSize(qtype GGUFType) int {
	switch qtype {
	case GGUFTypeF32, GGUFTypeF16:
		return 1
	case GGUFTypeQ4_0, GGUFTypeQ4_1, GGUFTypeQ5_0, GGUFTypeQ5_1, GGUFTypeQ8_0, GGUFTypeQ8_1:
		return 32
	case GGUFTypeQ2_K, GGUFTypeQ3_K, GGUFTypeQ4_K, GGUFTypeQ5_K, GGUFTypeQ6_K:
		return 256
	default:
		return 0
	}
}

func GGUFTypeQuantSize(qtype GGUFType) int {
	switch qtype {
	case GGUFTypeF32:
		return 4
	case GGUFTypeF16:
		return 2
	case GGUFTypeQ4_0:
		return 18
	case GGUFTypeQ4_1:
		return 20
	case GGUFTypeQ5_0:
		return 22
	case GGUFTypeQ5_1:
		return 24
	case GGUFTypeQ8_0:
		return 34
	case GGUFTypeQ8_1:
		return 40
	case GGUFTypeQ2_K:
		return 84
	case GGUFTypeQ3_K:
		return 110
	case GGUFTypeQ4_K:
		return 144
	case GGUFTypeQ5_K:
		return 176
	case GGUFTypeQ6_K:
		return 210
	default:
		return 0
	}
}

func GGUFTypeRowSize(qtype GGUFType, n int) int {
	bs := GGUFTypeBlockSize(qtype)
	qs := GGUFTypeQuantSize(qtype)
	if bs == 0 || qs == 0 {
		return 0
	}
	return (n / bs) * qs
}

// Fused dot products

func VecDotF32F32(src []byte, x []float32, n int) float32 {
	var sum float32
	for i := 0; i < n; i++ {
		w := math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
		sum += w * x[i]
	}
	return sum
}

func VecDotQ4KF32(src []byte, x []float32, n int) float32 {
	nb := n / 256
	var sumf float32
	offset := 0

	for i := 0; i < nb; i++ {
		d := fp16ToFp32(binary.LittleEndian.Uint16(src[offset:]))
		dmin := fp16ToFp32(binary.LittleEndian.Uint16(src[offset+2:]))
		scales := src[offset+4 : offset+16]
		qs := src[offset+16 : offset+144]
		xp := x[i*256:]

		is := 0
		qOff := 0
		for j := 0; j < 4; j++ {
			var sc, mn byte
			getScaleMinK4(is, scales, &sc, &mn)
			d1 := d * float32(sc)
			m1 := dmin * float32(mn)
			getScaleMinK4(is+1, scales, &sc, &mn)
			d2 := d * float32(sc)
			m2 := dmin * float32(mn)

			var sumQx1, sumX1, sumQx2, sumX2 float32
			for l := 0; l < 32; l++ {
				xLo := xp[l]
				xHi := xp[l+32]
				sumQx1 += float32(qs[qOff+l]&0xF) * xLo
				sumX1 += xLo
				sumQx2 += float32(qs[qOff+l]>>4) * xHi
				sumX2 += xHi
			}

			sumf += d1*sumQx1 - m1*sumX1 + d2*sumQx2 - m2*sumX2
			xp = xp[64:]
			qOff += 32
			is += 2
		}
		offset += 144
	}
	return sumf
}

func VecDotQ6KF32(src []byte, x []float32, n int) float32 {
	nb := n / 256
	var sumf float32
	offset := 0

	for i := 0; i < nb; i++ {
		d := fp16ToFp32(binary.LittleEndian.Uint16(src[offset+208:]))
		ql := src[offset : offset+128]
		qh := src[offset+128 : offset+192]
		sc := make([]int8, 16)
		for j := 0; j < 16; j++ {
			sc[j] = int8(src[offset+192+j])
		}
		xp := x[i*256:]

		var sums [16]float32

		for chunk := 0; chunk < 2; chunk++ {
			is := chunk * 8
			qlOff := chunk * 64
			qhOff := chunk * 32
			xpOff := chunk * 128

			for l := 0; l < 16; l++ {
				q1 := int((ql[qlOff+l]&0xF)|((qh[qhOff+l]>>0)&3)<<4) - 32
				q2 := int((ql[qlOff+l+32]&0xF)|((qh[qhOff+l]>>2)&3)<<4) - 32
				q3 := int((ql[qlOff+l]>>4)|((qh[qhOff+l]>>4)&3)<<4) - 32
				q4 := int((ql[qlOff+l+32]>>4)|((qh[qhOff+l]>>6)&3)<<4) - 32
				sums[is+0] += float32(q1) * xp[xpOff+l]
				sums[is+2] += float32(q2) * xp[xpOff+l+32]
				sums[is+4] += float32(q3) * xp[xpOff+l+64]
				sums[is+6] += float32(q4) * xp[xpOff+l+96]
			}
			for l := 16; l < 32; l++ {
				q1 := int((ql[qlOff+l]&0xF)|((qh[qhOff+l]>>0)&3)<<4) - 32
				q2 := int((ql[qlOff+l+32]&0xF)|((qh[qhOff+l]>>2)&3)<<4) - 32
				q3 := int((ql[qlOff+l]>>4)|((qh[qhOff+l]>>4)&3)<<4) - 32
				q4 := int((ql[qlOff+l+32]>>4)|((qh[qhOff+l]>>6)&3)<<4) - 32
				sums[is+1] += float32(q1) * xp[xpOff+l]
				sums[is+3] += float32(q2) * xp[xpOff+l+32]
				sums[is+5] += float32(q3) * xp[xpOff+l+64]
				sums[is+7] += float32(q4) * xp[xpOff+l+96]
			}
		}

		for j := 0; j < 16; j++ {
			sumf += d * float32(sc[j]) * sums[j]
		}
		offset += 210
	}
	return sumf
}

func VecDot(src []byte, x []float32, n int, qtype GGUFType) float32 {
	switch qtype {
	case GGUFTypeQ4_K:
		return VecDotQ4KF32(src, x, n)
	case GGUFTypeQ6_K:
		return VecDotQ6KF32(src, x, n)
	case GGUFTypeF32:
		return VecDotF32F32(src, x, n)
	default:
		buf := make([]float32, n)
		DequantizeRow(src, buf, n, qtype)
		return VecDotF32F32(nil, buf, n)
	}
}
