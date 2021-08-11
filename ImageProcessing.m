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
      matImage = zeros(sizeImage(1)*sizeImage(3),3);
      for i = 1:sizeImage(2) % Recorre las columnas de la imagen
        matImage(i:i+sizeImage(1)-1,:) = squeeze(img(:,i,:));
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

  end
end
 % Falta recorrer todas las imágenes, obtener su represetnación en matriz, agrupar estas representaciones en una sola matriz de (k x 3) y aplicar la cuatnización