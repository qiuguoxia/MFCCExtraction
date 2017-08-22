package com.lenovo.ca.AcousticClassification.test;

/**
 * Created by fubo5 on 2017/3/20.
 */
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.SourceDataLine;
import javax.sound.sampled.TargetDataLine;
import java.util.Arrays;


public class MicSoundTest
{
    private  TargetDataLine targetLine ;
    private byte[] targetData;
    private static final MicSoundTest testMic = new MicSoundTest();
    private MicSoundTest(){

    AudioFormat format = new AudioFormat(8000, 16, 1, true, true);
    DataLine.Info targetInfo = new DataLine.Info(TargetDataLine.class, format);
//        DataLine.Info sourceInfo = new DataLine.Info(SourceDataLine.class, format);

        try {
            targetLine = (TargetDataLine) AudioSystem.getLine(targetInfo);
            targetLine.open(format);
            targetLine.start();

//            SourceDataLine sourceLine = (SourceDataLine) AudioSystem.getLine(sourceInfo);
//            sourceLine.open(format);
//            sourceLine.start();


            System.out.println(targetLine.getBufferSize() / 5);
              targetData = new byte[targetLine.getBufferSize() / 5];
            //System.out.println(Arrays.toString(targetData));
        }
            catch (Exception e) {
                System.err.println(e);
            }
    }

    public static MicSoundTest getInstance(){
        return testMic;
    }

    public static void main(String[] args) {
        MicSoundTest testMic = MicSoundTest.getInstance();
        testMic.read();
        }

public void read(){
    int numBytesRead;
    int i = 0;
        numBytesRead = targetLine.read(targetData, 0, targetData.length);
        //if (numBytesRead == -1)	break;
        System.out.println(Arrays.toString(Arrays.copyOfRange(targetData, 0, 10)));
//                sourceLine.write(targetData, 0, numBytesRead);
}
}