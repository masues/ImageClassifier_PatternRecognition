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

    %{
    Crea un vector a partir de un tensor que representa a una imagen RGB.
    Args:
      img: imágen en formato tensor uint8 de tamaño (m x n x k)
      Como es una imagen rgb, k = 3
    Returns:
      Vector columna de tamaño mnk que representa todas las columnas de la
      imagen agrupadas en una sola, primero los valores R, luego los valores G
      y al último los valores B
    %}
    function vecImg = img2vec(img)
      sizeImage = size(img);
      mnk = sizeImage(1)*sizeImage(2)*sizeImage(3);
      % Si es posible, utiliza gpu para acelerar el proceso
      if gpuDeviceCount("available") > 0
        vecImg = gpuArray(reshape(img,[mnk 1]));
      else
        vecImg = reshape(img,[mnk 1]);
      end
    end

    %{
    Genera una matriz que representa a todas las muestras de imágenes.
    Esta función es utilizada para generar el conjunto de datos de entrada de la
    red neuronal.
    Args:
      imgs: Cell array que contiene clases de imágenes.
        La primer dimensión es la clase y la segunda se refiere a la imagen
        n-ésima. Todas las imagenes deben tener el mismo tamaño (m x n x k).
    Returns:
      Matriz de tamaño (mnk x l) que representa a todas las imágenes agrupadas
      como vecotores columna. l es el numero total de imágenes
    %}
    function imageInputs = imageInputs(imgs)
      imageInputsCell = {};
      % Itera sobre las clases de imagenes
      for i = 1:length(imgs)
        imageInputsCell{i} =  cell2mat( ...
          cellfun(@(img) ImageProcessing.img2vec(img), ...
          imgs{i}, 'UniformOutput', false) ...
        );
      end
      imageInputs = double(cell2mat(imageInputsCell));
    end

    %{
    Genera una matriz que representa la clasificación deseada dada un
    conjunto de imágenes.
    Args:
      imgs: Cell array que contiene clases de imágenes.
        La primer dimensión es la clase y la segunda se refiere a la imagen
        n-ésima.
    Returns:
      Matriz de tamaño (u x l) que representa a la clasificación de de los
      datos de entrada. u es el número de clases, l es el núemro de imágenes.
    %}
    function imageOutputs = imageOutputs(imgs)
      numClasses = length(imgs);
      lenClasses = [];
      % Itera sobre las clases de imagenes
      for i = 1:numClasses
        lenClasses = [ lenClasses , length(imgs{i})];
      end

      numOutputs = sum(lenClasses);

      % Crea una matriz con ceros
      imageOutputs = zeros([numClasses numOutputs]);

      % Llena las columnas correspondientes a la primer clase
      imageOutputs(1,1:lenClasses(1)) = ones(1,lenClasses(1));

      % Llena con 1's el renglón iésimo de los registros que corresponen a la
      % clase iésima
      for i = 2:numClasses
        offset = sum(lenClasses(1:i-1));
        imageOutputs(i,offset+1:offset+lenClasses(i)) = ones(1,lenClasses(i));
      end
    end

    %{
    Estandariza el tamaño de las imágenes
    Args:
      imgs: Cell array que contiene clases de imágenes.
        La primer dimensión es la clase y la segunda se refiere a la imagen
        n-ésima. Las imágenes son tensores uint8 de tamaño (m x n x k)
        Como es una imagen rgb, k = 3
      rows: Altura requerida para las imagenes
      cols: Anchura requerida para las imagenes
    Returns:
      Cell array que contiene clases de imágenes.
        La primer dimensión es la clase y la segunda se refiere a la imagen
        n-ésima estandarizada a (rows x cols) pixeles
    %}
    function standarizedImgs  = standarizeImgSizes(imgs,rows,cols)
      standarizedImgs = {};
      % Itera sobre las clases de imagenes
      for i = 1:length(imgs)
        standarizedImgs{i} = cellfun(@(img) imresize(img,[rows cols]), ...
          imgs{i}, 'UniformOutput', false);
      end
    end

  end
end
