# Copyright (c) 2022 Qilu University of Technology, School of Computer Science & Technology, 
  # Software Engineering(Software Development), Duyu (No.202103180009).
# version 3.0


print("[STEPS]Loading Basic Modules...")
# Basic function and constant.
from numpy import append as npappend
from numpy import zeros as npzeros
from numpy import float64 as npfloat64
from numpy import float32 as npfloat32
from numpy import int16 as npint16
from numpy import int32 as npint32
from numpy import array as nparray
from numpy import linspace as nplinspace


def getTypeFactor(dataType):
    typeFactor = 2 ** 15
    if dataType == npint16:
        typeFactor = 2 ** 15
    elif dataType == npint32:
        typeFactor = 2 ** 31
    elif dataType == npfloat32:
        typeFactor = 1
    return typeFactor


def reverb_funDefault(musicDataArr, volumeDb, offsetSample):
    k = 10.0 ** (-abs(volumeDb) / 20.0)
    return npappend(npzeros(offsetSample, npfloat64), musicDataArr * k)[:len(musicDataArr)]


def readPcmWavData(inputWavFileName_Original):
    print("[STEPS]Reading WAV data...")
    from scipy.io.wavfile import read as wavRead
    try:
        SampleRate_Original, MusicData = wavRead(inputWavFileName_Original)
    except Exception:
        from sys import stderr
        from sys import exit
        print("[ERROR]Read WAV file failed.",file=stderr)
        exit(1)
    left = []
    right = []
    for item in MusicData:
        left.append(item[0])
        right.append(item[1])
    Larr, Rarr = nparray(left), nparray(right)
    if not Larr.dtype == Rarr.dtype:
        raise Exception
    else:
        dataType = Larr.dtype
    left = Larr * 1.0
    right = Rarr * 1.0
    return SampleRate_Original, left, right, dataType


def writePcmWavData(outputWavFileName, left, right, SampleRate, dataType, sampleRate_Factor=0, isFilter=False, FilterDistance=7):
    print("[STEPS]Writing WAV data...")
    from scipy.io.wavfile import write as wavWrite
    from numpy import vstack as npvstack
    if not sampleRate_Factor == 0:
        left = left[range(0, len(left) - 1, sampleRate_Factor)]
        right = right[range(0, len(right) - 1, sampleRate_Factor)]
    else:
        sampleRate_Factor = 1
    if isFilter:
        print("[STEPS]Filtering...")
        from scipy.signal import medfilt
        medfilt(left, FilterDistance)
        medfilt(right, FilterDistance)
    MixedData = npvstack((left.astype(dataType), right.astype(dataType))).T
    try:
        wavWrite(outputWavFileName, int(SampleRate / sampleRate_Factor), MixedData)
    except Exception:
        from sys import stderr
        from sys import exit
        print("[ERROR]Read WAV file failed.",file=stderr)
        exit(1)


def reverb(SampleRate, left, right, echoFunction=reverb_funDefault, numberOfEcho=3, maxvolumeDb=2, offsetSecond=0.22,
           CONSTANT_REVERB_RANGE=12.0):
    print("[STEPS]Reverb Processing...")
    from numpy import divide as npdivide
    offsetSample = int(offsetSecond * SampleRate)
    sum = 0.0
    factorArr = nplinspace(maxvolumeDb, maxvolumeDb * CONSTANT_REVERB_RANGE, numberOfEcho)
    for a in factorArr:
        sum += 10.0 ** (-abs(a) / 20.0)
    left = npdivide(left, (sum + 1.0))
    right = npdivide(right, (sum + 1.0))
    c = 1
    for b in factorArr:
        left += echoFunction(musicDataArr=left, volumeDb=b, offsetSample=offsetSample * c)
        right += echoFunction(musicDataArr=right, volumeDb=b, offsetSample=offsetSample * c)
        ++c
    return left, right


def mixer(left, right, left_leftRate=1.0, left_rightRate=-1.0, right_leftRate=-1.0, right_rightRate=1.0):
    print("[STEPS]Mixing...")
    from numpy import add as npadd
    left_out = npadd(left * left_leftRate, right * left_rightRate)
    right_out = npadd(left * right_leftRate, right * right_rightRate)
    return left_out, right_out


# mode='Factor' or 'DB'
def gain(left, right, leftFactor=1.0, rightFactor=1.0, leftDB=0.0, rightDB=0.0, mode='Factor'):
    print("[STEPS]Gaining...")
    if mode.upper() == 'DB':
        leftFactor = 10.0 ** (-abs(leftDB) / 20.0)
        rightFactor = 10.0 ** (-abs(rightDB) / 20.0)
    return left * leftFactor, right * rightFactor


# This function may not work well. And it takes a long time.
def changeSpeed(left, right, dataType, frame_SampleLength=100):
    out_left = []
    out_right = []
    for i in range(0,len(left),frame_SampleLength):
        out_left.extend(left[i:i+frame_SampleLength])
        out_left.extend(npzeros(1, dataType))
    for i in range(0,len(right),frame_SampleLength):
        out_right.extend(left[i:i+frame_SampleLength])
        out_right.extend(npzeros(1, dataType))
    return nparray(out_left), nparray(out_right)


def showFrequencyDomainWave(sampleRate, waveData, dataType, fftSize=2048, offset=0, figSize=(8,4)):
    print("[STEPS]Analysing Frequency Domain Wave...")
    from pylab import plot
    from pylab import figure
    from pylab import xlabel
    from pylab import ylabel
    from pylab import show
    from numpy import log10 as nplog10
    from numpy import clip as npclip
    from numpy import abs as npabs
    from numpy.fft import rfft as nprfft
    typeFactor = getTypeFactor(dataType)
    xs = waveData[offset:fftSize + offset] / typeFactor
    xf = nprfft(xs) / fftSize
    freq = nplinspace(0, int(sampleRate / 2), int(fftSize / 2) + 1)
    xfp = 20 * nplog10(npclip(npabs(xf), 1e-20, 1e100))
    figure("Frequency Domain Wave", figsize=figSize)
    plot(freq, xfp)
    xlabel("Frequency(Hz)")
    ylabel("Volume(dB)")
    show()


def showTimeDomainWave(sampleRate, waveData, dataType, figSize=(8,4), xLabelUnit='second'):
    print("[STEPS]Analysing Time Domain Wave...")
    from pylab import plot
    from pylab import figure
    from pylab import xlabel
    from pylab import ylabel
    from pylab import show
    length = len(waveData)
    typeFactor = getTypeFactor(dataType)
    figure("Time Domain Wave", figsize=figSize)
    if xLabelUnit.lower() == 'second':
        plot(nparray(range(0, length)) / sampleRate, waveData / typeFactor)
        xlabel("Time(s)")
    else:
        plot(range(0, length), waveData / typeFactor)
        xlabel("Samples")
    ylabel("Amplitude")
    show()


def resample(original_sampleRate, current_sampleRate, resample_left, resample_right):
    print("[STEPS]Resample...")
    from numpy import interp as npinterp
    from numpy import arange as nparange
    factor = original_sampleRate * 1.0/current_sampleRate
    resample_left = npinterp(nparange(0, len(resample_left), factor), nparange(0, len(resample_left)), resample_left)
    resample_right = npinterp(nparange(0, len(resample_right), factor), nparange(0, len(resample_right)), resample_right)
    return current_sampleRate, resample_left, resample_right


# showMode=0:only print copyright information.
# showMode=1:only print usage.
# showMode=2:print both.
def showInformation(showMode):
    if showMode == 0 or showMode == 2:
        print("DuyuAudioProcessor-CORE v3.0")
        print("Copyright (c) 2022 Qilu University of Technology, School of Computer Science & Technology, "
              "Duyu (No.202103180009).")
    if showMode == 1 or showMode == 3:
        print("Usage of DuyuPCMprocessor-CORE")
        print()
        # Writing usage there.


# UNIT = second or sample.
def trim(sampleRate, start, end, left, right, UNIT="second"):
    print("[STEPS]Trim...")
    startSample = start
    endSample = end
    if UNIT.lower() == "second":
        startSample = start * sampleRate
        endSample = end * sampleRate
    try:
        return left[startSample:endSample+1], right[startSample:endSample+1]
    except Exception:
        from sys import stderr
        from sys import exit
        print("[ERROR]Can not trim.", file=stderr)
        exit(1)


def joint(left01, left02, right01, right02, sampleRate01, sampleRate02, out_sampleRate, dataType01, dataType02,
          out_dataType):
    print("[STEPS]Jointing...")
    currentSampleRate01, left01, right01 = resample(sampleRate01, out_sampleRate, left01, right01)
    left01, right01, currentDataType01 = changeBitRate(left01, right01, dataType01, out_dataType)
    currentSampleRate02, left02, right02 = resample(sampleRate02, out_sampleRate, left02, right02)
    left02, right02, currentDataType02 = changeBitRate(left02, right02, dataType02, out_dataType)
    out_left = npappend(left01, left02)
    out_right = npappend(right01, right02)
    return out_sampleRate, out_left, out_right, out_dataType


def changeBitRate(left, right, original_dataType, current_dataType):
    print("[STEPS]Changing bit rate...")
    left = (left.astype(npfloat64)/getTypeFactor(original_dataType))*getTypeFactor(current_dataType)
    right = (right.astype(npfloat64)/getTypeFactor(original_dataType))*getTypeFactor(current_dataType)
    return left, right, current_dataType
  

# test
if __name__ == '__main__':
    from warnings import filterwarnings
    from sys import argv
    filterwarnings('ignore')
    CommandLineArguments = argv
    inputWavFileName01 = r"C:\Users\35834\Desktop\CORE.wav"
    inputWavFileName02 = r"C:\Users\35834\Desktop\CORE2.wav"
    outputWavFileName = r"C:\Users\35834\Desktop\CORE2_OUT.wav"
    SampleRate, left, right, dataType = readPcmWavData(inputWavFileName01)
    SampleRate02, left02, right02, dataType02 = readPcmWavData(inputWavFileName02)
    SampleRate, left, right, dataType = joint(left, left02, right, right02, SampleRate, SampleRate02, 90000, dataType, dataType02, npint32)
    # left, right = reverb(SampleRate, left, right)
    # left, right = mixer(left, right)
    # left, right = gain(left, right, 2, 2)
    # left, right = changeSpeed(left, right, 16, 8)
    # showFrequencyDomainWave(SampleRate, left, dataType, offset=900000)
    # showTimeDomainWave(SampleRate, left, dataType)
    # left, right = changeSpeed(left, right, dataType)
    # SampleRate, left, right = resample(SampleRate, 94321, left, right)
    # showInformation(0)
    # left, right = trim(SampleRate, 20, 30, left, right)
    # left, right, dataType = changeBitRate(left, right, dataType, npfloat32)
    writePcmWavData(outputWavFileName, left, right, SampleRate, dataType, isFilter=True)
    print("[STEPS]Completed.")


# 2022/07/05 v2.0
# 2022/07/11 v3.0
