package com.lenovo.ca.AcousticClassification.utils;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

public class FileUtils {

	/**
	 * 写入文件
	 * 
	 * @param filepath
	 * @param content
	 */
	public static void WritetoFile(String filepath, String content) {
		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filepath)));
			bw.write(content);
			bw.newLine();
			bw.flush();
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (bw != null) {
				try {
					bw.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	/**
	 * Read file according encode
	 * 
	 * @param file
	 *            target file
	 * @param encoding
	 *            encoding of file
	 * @return content of file
	 */
	public static String ReadFile(File file, String encoding) {
		if (!file.exists())
			return null;

		BufferedReader reader = null;

		try {
			StringBuilder fileContent = new StringBuilder();
			InputStreamReader read = new InputStreamReader(new FileInputStream(file), encoding);
			reader = new BufferedReader(read);

			String tempString = null;

			while ((tempString = reader.readLine()) != null) {
				fileContent = fileContent.append("\r\n" + tempString);
			}

			return fileContent.toString();
		} catch (IOException e) {
			return null;
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {
					return null;
				}
			}
		}
	}

	/**
	 * 读取文件每一行到一个List对象中
	 * 
	 * @param filepath
	 * @return
	 */
	public static List<String> readLines(String filepath) {
		List<String> reList = new ArrayList<String>();
		File file = new File(filepath);
		if (!file.exists())
			return null;
		BufferedReader reader = null;
		try {
			// System.out.println("以行为单位读取文件内容，一次读一整行：");
			InputStreamReader read = new InputStreamReader(new FileInputStream(file), "UTF-8");
			reader = new BufferedReader(read);
			String tempString = null;
			// 一次读入一行，直到读入null为文件结束
			while ((tempString = reader.readLine()) != null) {
				// 显示行号
				// System.out.println("line " + line + ": " + tempString);
				reList.add(tempString);
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e1) {
				}
			}
		}
		return reList;
	}

	/**
	 * 写入文件
	 * 
	 * @param filepath
	 *            文件路径
	 * @param content
	 *            内容
	 * @param append
	 *            是否追加写入，True追加，False覆盖写入
	 * @param pageCode
	 *            编码
	 */
	public static void WriteFile(String filepath, String content, Boolean append, String pageCode) {
		if (content == null) {
			return;
		}
		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filepath, append), pageCode));
			bw.write(content);
			bw.newLine();
			bw.flush();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (bw != null) {
				try {
					bw.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	/**
	 * 批量写入到文件
	 * 
	 * @param filepath
	 * @param contents
	 * @param append
	 * @param pageCode
	 */
	public static void WriteFile(String filepath, List<String> contents, Boolean append, String pageCode) {
		if (null == null || contents.size() <= 0) {
			return;
		}
		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filepath, append), pageCode));
			for (int i = 0; i < contents.size(); i++) {
				bw.write(contents.get(i));
				bw.newLine();
				bw.flush();
			}

		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (bw != null) {
				try {
					bw.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}
}
