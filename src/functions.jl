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
    _function_to_declaration(tool::Function) -> FunctionDeclaration

Build a minimal `FunctionDeclaration` from a plain Julia function, using the
function's name and an empty object parameter schema. For typed schemas, pass
a `FunctionDeclaration` or a `Dict` instead.
"""
function _function_to_declaration(tool::Function)
    return FunctionDeclaration(;
        name=string(nameof(tool)),
        description=nothing,
        parameters=FunctionParameter(; type="object"),
    )
end

function _build_tools(config::GenerateContentConfig)
    tools = Dict{String,Any}[]
    declarations = Dict{String,Any}[]
    for tool in something(config.tools, ())
        if tool isa Function
            push!(declarations, to_api_function_declaration(_function_to_declaration(tool)))
        elseif tool isa AbstractDict
            push!(tools, string_key_dict(tool))
        else
            @warn "Ignoring unsupported tool type: $(typeof(tool))"
        end
    end
    for fd in something(config.function_declarations, ())
        push!(declarations, to_api_function_declaration(fd))
    end
    isempty(declarations) ||
        push!(tools, Dict{String,Any}("functionDeclarations" => declarations))
    return tools
end

function _build_tool_config(config::GenerateContentConfig)
    config.tool_config === nothing && return nothing
    fc = config.tool_config.function_calling_config
    fc === nothing && return nothing
    fc_dict = Dict{String,Any}("mode" => fc.mode)
    fc.allowed_function_names !== nothing &&
        (fc_dict["allowedFunctionNames"] = fc.allowed_function_names)
    return Dict{String,Any}("functionCallingConfig" => fc_dict)
end

"""
    string_to_symbol_keys(dict::AbstractDict) -> Dict{Symbol, Any}

Convert a dictionary with string keys to one with symbol keys.
This is needed for Julia keyword arguments which expect symbols.
"""
function string_to_symbol_keys(dict::AbstractDict)
    return Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(dict))
end

function _call_function(
    functions::AbstractDict{String,<:Function}, name::AbstractString, args
)
    haskey(functions, name) ||
        return Dict{String,Any}("error" => "Function $name not found")
    result = try
        functions[name](; string_to_symbol_keys(args)...)
    catch e
        return Dict{String,Any}("error" => "Error executing function: $e")
    end
    result === nothing && return Dict{String,Any}("status" => "completed")
    result isa AbstractDict && return result
    return Dict{String,Any}("value" => result)
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
    return Dict{String,Any}(
        fc.name => _call_function(functions, fc.name, fc.args) for fc in function_calls
    )
end
