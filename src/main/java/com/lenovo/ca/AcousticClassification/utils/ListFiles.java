package com.lenovo.ca.AcousticClassification.utils;


import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;

/**
 * Created by fubo5 on 2017/3/31.
 */
public class ListFiles {
    /**
     * Load all the file names in the given folder not the sub folders.
     * @param directoryName The folder to find the files
     * @param extension The extension name of which the files are loaded
     * @return
     */
    public static File[] currentDir(String directoryName, final String extension) {
        // Load all .wav files in the folder
        File dir = new File(directoryName);
        File [] files = dir.listFiles(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.endsWith(extension);
            }
        });
        return files;
    }

    /**
     * Load all files not only in the current folder but also its sub folders
     * @param directoryName
     * @param files The file names to return
     * @param extension
     */
    public static void allDirs(String directoryName, ArrayList<File> files, String extension) {
        File directory = new File(directoryName);

        // get all the files from a directory
        File[] fList = directory.listFiles();
        for (File file : fList) {
            if (file.isFile()) {
                if (file.getName().endsWith(extension)) files.add(file);
            } else if (file.isDirectory()) {
                allDirs(file.getAbsolutePath(), files, extension);
            }
        }
    }

    /**
     * Overload the allDirs method, when no extension is specified, load all files
     * @param directoryName
     * @param files
     */
    public static void allDirs(String directoryName, ArrayList<File> files) {
        allDirs(directoryName, files, "");
    }

    public static void main(String[] args) {
        File[] files = currentDir("D:\\Projects\\AcousticClassification" +
                "\\Data\\DIY\\Music", ".wav");
        //for (File file : files) System.out.println(file);
        ArrayList<File> maleFiles = new ArrayList<File>();
        allDirs("D:\\Projects\\AcousticClassification\\Data\\VoxForge\\train\\male", maleFiles);
        for (File mFile: maleFiles) System.out.println(mFile);
    }
}
