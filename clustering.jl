module ShipClustering

using DataFrames
using Statistics
using Random

export distance_to_edges, closest_edge,
    find_centroid, pop_rand!, score_cluster, sort_clusters, score_clustering, find_clusters_greedy,
    find_clusters, get_clusters_with_constraints, get_targeted_edges, get_targets, RADIUS_DAMAGE

const RADIUS_DAMAGE = 20 # радиус эффективности оружия
const NUM_TARGETS = 2 # кол-во целей, которые можно поразить за 1 момент
const N_ROLLS = 30 # кол-во повторений для поиска лучшией кластеризации

"""
Функция для определения дистанции от точки до всех остальных точек в матрице
"""
function distance_to_edges(coords, edge_val)
    diff_edge = coords .- edge_val
    euclidean_dist = sqrt.(vec(sum(diff_edge .^ 2, dims=2)))
    euclidean_dist
end

"""
Определение ближайшей точки и растояния до неё.
indices позволяет отфильтровать точки из матрицы
"""
function closest_edge(coords, edge_val, indices)
    closest_distance = Inf
    closest_edge_val = 0
    for i in indices
        euclidean_dist = sqrt(sum((coords[i, :] .- edge_val) .^ 2))
        if euclidean_dist < closest_distance
            closest_distance = euclidean_dist
            closest_edge_val = i
        end
    end
    if closest_edge_val == 0
        error("no available indices")
    end
    closest_edge_val, closest_distance
end

"Определение растояния без фильтра по строкам"
function closest_edge(coords, edge)
    euclidean_dist = distance_to_edges(coords, edge)
    euclidean_dist[edge] = Inf
    argmin(euclidean_dist)
end

"Найти центроиду по матрице с точками"
function find_centroid(coords)
    mean(coords, dims=1)
end

"Выбросить случайный элемент из сета"
function pop_rand!(s::Set)
    pop!(s, rand(s))
end

"""
Оценить качествo найденного кластера.
На выходе tuple(кол-во элементов, -сумма растояний от центродиы)
То есть чем больше этот tuple, тем для нас лучше
"""
function score_cluster(coords, cluster)
    center = find_centroid(coords[cluster, :])
    distances_to_members = distance_to_edges(coords[cluster, :], center)
    length(cluster), -sum(distances_to_members)
end

"""
Отсортировать кластеры по их полезности
"""
function sort_clusters(coords, clusters)
    sorted_dict = sort(collect(clusters), by=x -> score_cluster(coords, x[2]), rev=true)
    return [v for (_, v) in sorted_dict]
end

"""
Получить общее значение качества на кластеризацию.
Собирается из качества индивидуальных лучших кластеров.
"""
function score_clustering(coords, clusters, num_targets)
    best_clusters = sort_clusters(coords, clusters)[begin:num_targets]
    scores_matrix = reduce(vcat, transpose.(map(x -> collect(score_cluster(coords, x)), best_clusters)))
    Tuple(sum(scores_matrix, dims=1))
end

"""
Жадно найти лучшие кластеры.
Присутствует элемент случайности.
"""
function find_clusters_greedy(coords, threshold)
    clusters = Dict{Int64,Vector{Int64}}()
    available_edges = Set([1:size(coords)[1];])
    current_cluster = 1
    clusters[current_cluster] = [pop_rand!(available_edges)]
    while !isempty(available_edges)
        cluster_center = find_centroid(coords[clusters[current_cluster], :])
        cl_edge_idx, cl_distance = closest_edge(coords, vec(cluster_center), available_edges)
        to_be_cluster = coords[vcat(clusters[current_cluster], [cl_edge_idx]), :]

        new_cluster_center = find_centroid(to_be_cluster)

        all_distances = distance_to_edges(to_be_cluster, new_cluster_center)

        # Или дистация до ближайшей точки выше порога
        # Или центроида сдвинется так, что какая-то старая точка выпадет из кластера
        if cl_distance > threshold || any(x -> x > threshold, all_distances)
            current_cluster += 1
            clusters[current_cluster] = [pop_rand!(available_edges)]
        else
            delete!(available_edges, cl_edge_idx)
            push!(clusters[current_cluster], cl_edge_idx)
        end
    end
    clusters
end

"""
Прогон жадной и случайной кластеризации несколько раз, чтобы найти лучшую кластеризацию.
"""
function find_clusters(coords, threshold; num_targets=NUM_TARGETS)
    best_clustering = Dict()
    best_quality = (-Inf, -Inf)
    for _ in 1:N_ROLLS
        clusters = find_clusters_greedy(coords, threshold)
        quality = score_clustering(coords, clusters, num_targets)
        if quality > best_quality
            best_clustering = clusters
            best_quality = quality
        end
    end
    sort_clusters(coords, best_clustering)
end

"""Получение лучших кластеров с указанными ограничениями"""
function get_clusters_with_constraints(coords; radius_damage=RADIUS_DAMAGE, num_targets=NUM_TARGETS)
    find_clusters(coords, radius_damage, num_targets=num_targets)[begin:num_targets]
end

"""Получить все точки, попавшие в лучшие кластера"""
function get_targeted_edges(clusters_constrained)
    Set(reduce(vcat, clusters_constrained))
end

"""Получить центроиды лучших кластеров, являющиеся целями."""
function get_targets(coords, clusters_constrained)
    center_targets = reduce(vcat, map(x -> find_centroid(coords[x, :]), clusters_constrained))
    center_targets
end

end