classdef ImageProcessing
  methods(Static)
    %{
    Obtiene todas las imágnes de un directorio dado y las almacena en un cell
    array
    Args:
      directory: Nombre del directorio en el que se encuentran las imágenes
      extension: Extensión de las imágenes
    Returns:
      Cell array con imágenes en formato tensor uint8 de tamaño (m x n x 3)
    %}
    function images = getImagesInDir(directory,extension)
        % Obtiene una lista de imágenes del directorio dado
        imageFiles = dir(directory+'/*'+extension);
        nFiles = length(imageFiles);
        for i = 1:nFiles
          currentFilename = imageFiles(i).name;
          images{i} = imread(directory+'/'+currentFilename);
        end
    end

    %{
    Obtiene las imágenes agrupadas en clases
    Se asume que cada subdirectorio del directorio dado posee imágenes de una
    sola clase.
    Args:
      directory: Nombre del directorio que contiene las clases
    Returns:
      imageClasses: Cell array que contiene clases de imágenes.
        La primer dimensión es la clase y la segunda se refiere a la imagen
        n-ésima.
      imageMap: container.Map para mapear los valores numéricos de las
        clasificaciones a los nombres de la clase
    %}
    function [imageClasses,imageMap] = getImagesClasses(directory)
      % Obtiene una lista de directorios
      files = dir(directory);
      nFiles = length(files);
      % Inicializa las llaves y valor del imagemap
      keySet = zeros(1,nFiles-2);
      valueSet = cell(1,nFiles-2);
      % Inicia en 3 porque los primeros directorios son '.' y '..'
      for i = 3:nFiles
        currentFilename = files(i).name;
        % Obtiene las imágenes
        imageClasses{i-2} = ImageProcessing.getImagesInDir(directory+"/"+currentFilename,".jpg");
        % Almacena la llave y su valor
        keySet(i-2) = i-2;
        valueSet{i-2} = currentFilename;
      end
      imageMap = containers.Map(keySet,valueSet);
    end


    %{
    Crea una matriz a partir de un tensor que representa a una imagen RGB.
    Args:
      img: imágen en formato tensor uint8 de tamaño (m x n x 3)
    Returns:
      Matriz de tamaño (mn x 3) que representa todas las columnas de la imagen
      agrupadas en una sola
    %}
    function matImage = img2matrix(img)
      sizeImage = size(img);
      mn = sizeImage(1)*sizeImage(2);
      k = sizeImage(3);
      % Si es posible, utiliza gpu para acelerar el proceso
      if gpuDeviceCount("available") > 0
        matImage = gpuArray(reshape(img,[mn k]));
      else
        matImage = reshape(img,[mn k]);
      end
    end


    %{
    Genera una matriz que representa a todos valores RGB de una clase de
    imágenes
    Args:
      img: Cell array con imágenes en formato tensor uint8 de tamaño (m x n x 3)
    Returns:
      Matriz de tamaño (k x 3) que representa a todos los pixeles RGB de todas
      las imágenes de la clase.
    %}
    function matClass = imgClass2matrix(imageClass)
      matClass = [];
      for i=1:length(imageClass)
        matClass = [matClass; ImageProcessing.img2matrix(imageClass{i})];
      end
    end

        %{
    Rescala la imagen en relación 2:1
    Aplica filtro de sobel horizontal y vertical
    Args:
      img: Cell array con imágenes en formato tensor uint8 de tamaño (m x n x 3)
    Returns:
      Matriz de tamaño (k x 3) que representa a todos los pixeles RGB de todas
      las imágenes de la clase.
    %}

    function img_filter = filter(image)
      sobel_V = [-1 0 1; -2 0 2; -1 0 1];
      sobel_H = transpose(sobel_V);
      r = image(:,:,1);
      g = image(:,:,2);
      b = image(:,:,3);
      r_v = conv2(r,sobel_V,'same');
      g_v = conv2(g,sobel_V,'same');
      b_v = conv2(b,sobel_V,'same');
      r_h = conv2(r,sobel_H,'same');
      g_h = conv2(g,sobel_H,'same');
      b_h = conv2(b,sobel_H,'same');
      img_filter = cat(3,r,g,b,r_v,g_v,b_v,r_h,g_h,b_h);
      %Al hacer la concatenación pasa los doubles de la convocuión a uint8
    end

  end
end
 % Falta recorrer todas las imágenes, obtener su represetnación en matriz, agrupar estas representaciones en una sola matriz de (k x 3) y aplicar la cuatnización
