classdef CuantizadorVectorial
  methods(Static)
    
    
    function [indx,centroides] = estabilizador(centroides, data)
      
      lenCent = size(centroides,1);
      distances = pdist2(data,centroides);
      [distMenor,indx] = min(distances, [ ], 2);
      distGAnterior = 0;
      distGActual = sum(distMenor);
      
      while abs(distGActual - distGAnterior) > 0.5
        for i = 1:lenCent
          ind = indx==i;
          dataGroup = data(ind,:);
          lenDG = size(dataGroup,1);
          centroides(i,:) = (1/lenDG(1)) * sum (dataGroup);
        end
        distances = pdist2(data,centroides);
        [distMenor,indx] = min(distances, [ ], 2);
        distGAnterior = distGActual;
        distGActual = sum(distMenor);
      end
    end
    
    
    
    function  [indx,centroides] = LindeBuzoGray(centroides, nCuant, data)
      nCentroides = [centroides * 0.9999; centroides * 1.0001]; %Nuevos centroides
      [indx,centroides] = CuantizadorVectorial.estabilizador(nCentroides, data);
      lenCent = size(centroides,1);
      if lenCent < nCuant
        [indx,centroides] = CuantizadorVectorial.LindeBuzoGray(centroides, nCuant, data);
      end
    end


    %{
    Obtiene la distancia de una observacion (conjunto de puntos)
    a un cuantizador vectorial.
    Args:
      data: Matriz que contiene a los vectores de la observación.
      centroidesCuantizador:  Matriz que contiene a los centroides del
        cuantizador vectorial.
    Returns:
      Distancia de la observación al cuantizador vectorial.
    %}
    function distGlobal = distCuantizador(data, centroidesCuantizador)
      distancias = pdist2(data,centroidesCuantizador);
      distMenor = min(distancias, [ ], 2);
      distGlobal = sum(distMenor);
    end

    %{
    Clasifica a una señal de voz utilizando los cuantizadores de entrada
    Args:
      data: Matriz que contiene a los vectores de la observación.
      cuantizadores:
        Cell array que representa a los cuantizadores vectoriales.
        La primer dimensión corresponde al cuantizador n-ésimo
        La segunda dimensión corresponde a
          1 -> Índice del grupo en el que se agrupó cada observación
          2 -> Centroides del cuantizador vectorial
    Returns:
      Índice del cuantizador vectorial en el que se clasificó la observación
    %}
    function indx = clasificador(data, cuantizadores)
      % Obtiene el número de cuantizadores
      numCuantizadores = length(cuantizadores);
      % Inicializa la distancia de la señal a cada cuantizador
      distCuantizadores = zeros(1,numCuantizadores);
      for i=1:numCuantizadores
        distCuantizadores(i) = CuantizadorVectorial.distCuantizador(data,cuantizadores{i}{2});
      end
      [~,indx] = min(distCuantizadores);
    end
    
  end
end
