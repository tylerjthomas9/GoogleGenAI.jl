"""
    FunctionDeclaration(name, description, parameters)

Create a function declaration with a Dict-based parameters format.
"""
function FunctionDeclaration(
    name::String, description::Union{String,Nothing}, parameters::Dict{String,Any}
)
    param_type = get(parameters, "type", "object")
    param_desc = get(parameters, "description", nothing)
    param_props = get(parameters, "properties", nothing)
    param_items = get(parameters, "items", nothing)
    param_required = get(parameters, "required", nothing)

    param = FunctionParameter(;
        type=param_type,
        description=param_desc,
        properties=param_props,
        items=param_items,
        required=param_required,
    )

    return FunctionDeclaration(name, description, param)
end

"""
    string_to_symbol_keys(dict::Dict{String, Any}) -> Dict{Symbol, Any}

Convert a dictionary with string keys to one with symbol keys.
This is needed for Julia keyword arguments which expect symbols.
"""
function string_to_symbol_keys(dict::Dict{String,Any})
    symbol_dict = Dict{Symbol,Any}()
    for (key, value) in dict
        symbol_dict[Symbol(key)] = value
    end
    return symbol_dict
end

"""
    build_function_conversation(user_query, function_name, function_args, function_result)

Builds a properly formatted conversation for function calling.
Returns a conversation vector ready for the final response.
"""
function build_function_conversation(
    user_query::String, function_name::String, function_args::Dict, function_result::Dict
)
    return [
        Dict(:role => "user", :parts => [Dict(:text => user_query)]),
        Dict(
            :role => "model",
            :parts => [
                Dict(:functionCall =>
                        Dict(:name => function_name, :args => function_args)),
            ],
        ),
        Dict(
            :role => "function",
            :parts => [
                Dict(
                    :functionResponse =>
                        Dict(:name => function_name, :response => function_result),
                ),
            ],
        ),
    ]
end

"""
    execute_parallel_function_calls(
        function_calls::Vector, 
        functions::Dict{String, <:Function}
    ) -> Dict{String, Any}

Execute multiple function calls in parallel and collect their results.

# Arguments
- `function_calls::Vector`: Function calls from the model (can be Any or FunctionCall)
- `functions::Dict{String, <:Function}`: Dictionary of available functions

# Returns
- `Dict{String, Any}`: Dictionary mapping function names to their results
"""
function execute_parallel_function_calls(
    function_calls::Vector, functions::Dict{String,<:Function}
)
    results = Dict{String,Any}()

    for function_call in function_calls
        name = function_call.name
        args = function_call.args

        if !haskey(functions, name)
            results[name] = Dict("error" => "Function $name not found")
            continue
        end

        try
            symbol_args = string_to_symbol_keys(args)
            result = functions[name](; symbol_args...)
            results[name] = if isa(result, Dict)
                result
            elseif result === nothing
                Dict("status" => "completed")
            else
                Dict("value" => result)
            end
        catch e
            results[name] = Dict("error" => "Error executing function: $e")
        end
    end

    return results
end
